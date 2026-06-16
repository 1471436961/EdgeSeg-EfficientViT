"""Replace EfficientViT relu_linear_att subgraphs with the Phase 3 Plugin.

This script performs only P1a graph surgery:

    Concat_output_0 -> Cast_1_output_0

for the selected LiteMLA context blocks. It keeps qkv, aggregation, concat,
proj, and residual add untouched.
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


SCRIPT_NAME = "integrate_relu_linear_attention_plugin_onnx.py"
PLUGIN_NAME = "EdgesegReluLinearAttention_TRT"
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "edgeseg"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
PHASE2_SCRIPTS = ROOT / "phase2" / "scripts"
if str(PHASE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PHASE2_SCRIPTS))

from _common import resolve_script_version, save_json, sha256_of_file, version_of  # noqa: E402


DEFAULT_ONNX = Path("phase2/results/onnx/efficientvit_seg_b0_cityscapes_1024x2048.onnx")
DEFAULT_PATCHED_ONNX = Path(
    "phase3/results/onnx/efficientvit_seg_b0_cityscapes_1024x2048_relu_linear_att_plugin.onnx"
)
DEFAULT_METADATA = Path("phase3/results/metrics/relu_linear_attention_plugin_onnx_integration.json")


STAGE2_TARGETS = [
    {
        "block_prefix": "/backbone/stages.2/op_list.1/context_module/main",
        "input_c": 384,
        "height": 64,
        "width": 128,
    },
    {
        "block_prefix": "/backbone/stages.2/op_list.2/context_module/main",
        "input_c": 384,
        "height": 64,
        "width": 128,
    },
]

STAGE3_TARGETS = [
    {
        "block_prefix": "/backbone/stages.3/op_list.1/context_module/main",
        "input_c": 768,
        "height": 32,
        "width": 64,
    },
    {
        "block_prefix": "/backbone/stages.3/op_list.2/context_module/main",
        "input_c": 768,
        "height": 32,
        "width": 64,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrate the relu_linear_att Plugin into the Phase 2 ONNX graph.")
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="Input Phase 2 ONNX path.")
    p.add_argument("--output", type=Path, default=DEFAULT_PATCHED_ONNX, help="Patched ONNX output path.")
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="Integration metadata JSON path.")
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--eps", type=float, default=1.0e-15)
    p.add_argument(
        "--target-scope",
        choices=("stage2", "stage2-stage3"),
        default="stage2",
        help="Which LiteMLA relu_linear_att blocks to replace.",
    )
    p.add_argument("--skip-checker", action="store_true", help="Skip ONNX checker.")
    return p.parse_args()


def target_specs(scope: str) -> List[Dict[str, Any]]:
    if scope == "stage2":
        return list(STAGE2_TARGETS)
    if scope == "stage2-stage3":
        return list(STAGE2_TARGETS) + list(STAGE3_TARGETS)
    raise ValueError(f"Unsupported target scope: {scope}")


def build_producer_map(nodes) -> Dict[str, Any]:
    producers: Dict[str, Any] = {}
    for node in nodes:
        for output in node.output:
            producers[output] = node
    return producers


def build_consumer_map(nodes) -> Dict[str, List[Any]]:
    consumers: Dict[str, List[Any]] = defaultdict(list)
    for node in nodes:
        for input_name in node.input:
            consumers[input_name].append(node)
    return consumers


def collect_reverse_subgraph(producers: Dict[str, Any], output_tensor: str, stop_tensor: str) -> Set[str]:
    """Collect producer node names from output_tensor back to stop_tensor."""
    remove: Set[str] = set()
    visited_tensors: Set[str] = set()

    def visit_tensor(tensor_name: str) -> None:
        if tensor_name == stop_tensor or tensor_name in visited_tensors:
            return
        visited_tensors.add(tensor_name)
        node = producers.get(tensor_name)
        if node is None:
            return
        remove.add(node.name)
        for input_name in node.input:
            visit_tensor(input_name)

    visit_tensor(output_tensor)
    return remove


def make_plugin_node(onnx, target: Dict[str, Any], args: argparse.Namespace):
    from onnx import helper

    block_prefix = str(target["block_prefix"])
    return helper.make_node(
        PLUGIN_NAME,
        inputs=[f"{block_prefix}/Concat_output_0"],
        outputs=[f"{block_prefix}/Cast_1_output_0"],
        name=f"{block_prefix}/EdgesegReluLinearAttention_TRT",
        plugin_version=PLUGIN_VERSION,
        plugin_namespace=PLUGIN_NAMESPACE,
        dim=int(args.dim),
        eps=float(args.eps),
        input_c=int(target["input_c"]),
        height=int(target["height"]),
        width=int(target["width"]),
    )


def remove_value_info_for_removed_tensors(model, removed_tensors: Set[str], preserved_tensors: Set[str]) -> int:
    names_to_remove = removed_tensors - preserved_tensors
    keep = [vi for vi in model.graph.value_info if vi.name not in names_to_remove]
    removed = len(model.graph.value_info) - len(keep)
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep)
    return removed


def replace_blocks(model, onnx, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], int]:
    original_nodes = list(model.graph.node)
    producers = build_producer_map(original_nodes)
    consumers = build_consumer_map(original_nodes)
    targets = target_specs(args.target_scope)

    remove_names: Set[str] = set()
    removed_tensors: Set[str] = set()
    insertions: Dict[int, Any] = {}
    block_meta: List[Dict[str, Any]] = []

    for target in targets:
        block_prefix = str(target["block_prefix"])
        plugin_input = f"{block_prefix}/Concat_output_0"
        plugin_output = f"{block_prefix}/Cast_1_output_0"

        if plugin_input not in consumers:
            raise RuntimeError(f"Plugin input has no consumer: {plugin_input}")
        if plugin_output not in producers:
            raise RuntimeError(f"Plugin output producer not found: {plugin_output}")

        block_remove = collect_reverse_subgraph(producers, plugin_output, plugin_input)
        if not block_remove:
            raise RuntimeError(f"No removable subgraph found for {block_prefix}")

        block_indices = [i for i, node in enumerate(original_nodes) if node.name in block_remove]
        insert_index = min(block_indices)
        plugin_node = make_plugin_node(onnx, target, args)
        insertions[insert_index] = plugin_node
        remove_names.update(block_remove)

        block_removed_nodes = [node for node in original_nodes if node.name in block_remove]
        for node in block_removed_nodes:
            removed_tensors.update(node.output)

        block_meta.append(
            {
                "block_prefix": block_prefix,
                "plugin_input": plugin_input,
                "plugin_output": plugin_output,
                "plugin_attrs": {
                    "dim": int(args.dim),
                    "eps": float(args.eps),
                    "input_c": int(target["input_c"]),
                    "height": int(target["height"]),
                    "width": int(target["width"]),
                },
                "insert_index": insert_index,
                "removed_node_count": len(block_removed_nodes),
                "removed_nodes": [
                    {"name": node.name, "op_type": node.op_type, "outputs": list(node.output)}
                    for node in block_removed_nodes
                ],
                "plugin_node": {
                    "name": plugin_node.name,
                    "op_type": plugin_node.op_type,
                    "inputs": list(plugin_node.input),
                    "outputs": list(plugin_node.output),
                },
            }
        )

    patched_nodes = []
    for index, node in enumerate(original_nodes):
        if index in insertions:
            patched_nodes.append(insertions[index])
        if node.name not in remove_names:
            patched_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(patched_nodes)
    removed_value_info = remove_value_info_for_removed_tensors(
        model,
        removed_tensors,
        preserved_tensors={f"{target['block_prefix']}/Cast_1_output_0" for target in targets},
    )
    return block_meta, removed_value_info


def run_checker(onnx, model, skip: bool) -> Dict[str, Any]:
    if skip:
        return {"status": "skipped", "error": None}
    try:
        onnx.checker.check_model(model)
        return {"status": "ok", "error": None}
    except Exception as exc:
        return {
            "status": "warning",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "note": "ONNX checker may not know TensorRT custom Plugin ops; parser build is the authoritative Step 6 check.",
        }


def integrate(args: argparse.Namespace) -> Dict[str, Any]:
    import onnx

    onnx_path = args.onnx.expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(onnx_path))
    original_node_count = len(model.graph.node)
    block_meta, removed_value_info = replace_blocks(model, onnx, args)
    checker = run_checker(onnx, model, args.skip_checker)
    onnx.save(model, str(output_path))

    plugin_node_count = sum(1 for node in model.graph.node if node.op_type == PLUGIN_NAME)
    payload = {
        "status": "ok",
        "purpose": "phase3_step6_relu_linear_attention_plugin_onnx_integration",
        "input_onnx": {
            "path": str(onnx_path),
            "sha256": sha256_of_file(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
        },
        "output_onnx": {
            "path": str(output_path),
            "sha256": sha256_of_file(output_path),
            "size_bytes": output_path.stat().st_size,
        },
        "plugin": {
            "op_type": PLUGIN_NAME,
            "plugin_version": PLUGIN_VERSION,
            "plugin_namespace": PLUGIN_NAMESPACE,
            "dim": int(args.dim),
            "eps": float(args.eps),
            "target_scope": args.target_scope,
            "target_specs": [
                {
                    "block_prefix": str(target["block_prefix"]),
                    "input_c": int(target["input_c"]),
                    "height": int(target["height"]),
                    "width": int(target["width"]),
                }
                for target in target_specs(args.target_scope)
            ],
        },
        "graph": {
            "original_node_count": original_node_count,
            "patched_node_count": len(model.graph.node),
            "plugin_node_count": plugin_node_count,
            "target_block_count": len(target_specs(args.target_scope)),
            "removed_value_info": removed_value_info,
        },
        "blocks": block_meta,
        "checker": checker,
        "versions": {
            "python": platform.python_version(),
            "onnx": onnx.__version__,
            "numpy": version_of("numpy"),
        },
        "notes": [
            "This is P1a relu_linear_att-only graph surgery.",
            "target_scope=stage2 keeps the original Phase 3 P1a behavior; target_scope=stage2-stage3 also replaces the smaller stage3 context blocks.",
            "It preserves qkv, aggregation, concat, proj, and residual add.",
            "Numerical correctness and latency are Step 7 responsibilities.",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }
    save_json(args.metadata.expanduser().resolve(), payload)
    return payload


def save_failure_metadata(args: argparse.Namespace, error: Exception) -> None:
    payload = {
        "status": "failed",
        "purpose": "phase3_step6_relu_linear_attention_plugin_onnx_integration",
        "target_scope": args.target_scope,
        "onnx": str(args.onnx.expanduser().resolve()),
        "output": str(args.output.expanduser().resolve()),
        "error_type": type(error).__name__,
        "error": str(error),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }
    save_json(args.metadata.expanduser().resolve(), payload)


def main() -> None:
    args = parse_args()
    try:
        payload = integrate(args)
    except Exception as exc:
        save_failure_metadata(args, exc)
        raise

    print(
        "Plugin ONNX integration complete: "
        f"output={payload['output_onnx']['path']} "
        f"metadata={args.metadata} "
        f"plugin_nodes={payload['graph']['plugin_node_count']} "
        f"nodes={payload['graph']['original_node_count']}->{payload['graph']['patched_node_count']}"
    )


if __name__ == "__main__":
    main()
