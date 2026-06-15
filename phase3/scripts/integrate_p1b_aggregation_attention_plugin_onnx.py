"""Replace EfficientViT stage2 aggregation + attention subgraphs with P1b Plugin.

This script performs P1b graph surgery:

    qkv/conv/Conv_output_0 -> Cast_1_output_0

for the two `backbone.stages.2` LiteMLA context blocks. It keeps qkv, proj,
and residual add outside the Plugin. Aggregation weights are passed to the
Plugin node as ONNX initializer inputs.
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


SCRIPT_NAME = "integrate_p1b_aggregation_attention_plugin_onnx.py"
PLUGIN_NAME = "EdgesegAggregationReluLinearAttention_TRT"
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
    "phase3/results/onnx/efficientvit_seg_b0_cityscapes_1024x2048_p1b_aggregation_attention_plugin.onnx"
)
DEFAULT_METADATA = Path("phase3/results/metrics/p1b_aggregation_attention_plugin_onnx_integration.json")


TARGET_BLOCKS = [
    {
        "block_prefix": "/backbone/stages.2/op_list.1/context_module/main",
        "initializer_prefix": "backbone.stages.2.op_list.1.context_module.main",
    },
    {
        "block_prefix": "/backbone/stages.2/op_list.2/context_module/main",
        "initializer_prefix": "backbone.stages.2.op_list.2.context_module.main",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrate the P1b aggregation + attention Plugin into Phase 2 ONNX.")
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="Input Phase 2 ONNX path.")
    p.add_argument("--output", type=Path, default=DEFAULT_PATCHED_ONNX, help="Patched ONNX output path.")
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="Integration metadata JSON path.")
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--eps", type=float, default=1.0e-15)
    p.add_argument("--qkv-c", type=int, default=192)
    p.add_argument("--output-c", type=int, default=128)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--skip-checker", action="store_true", help="Skip ONNX checker.")
    return p.parse_args()


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


def initializer_shapes(model) -> Dict[str, List[int]]:
    return {init.name: [int(dim) for dim in init.dims] for init in model.graph.initializer}


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


def make_plugin_node(onnx, block_prefix: str, initializer_prefix: str, args: argparse.Namespace):
    from onnx import helper

    qkv_tensor = f"{block_prefix}/qkv/conv/Conv_output_0"
    plugin_output = f"{block_prefix}/Cast_1_output_0"
    dw_weight = f"{initializer_prefix}.aggreg.0.0.weight"
    pw_weight = f"{initializer_prefix}.aggreg.0.1.weight"

    return helper.make_node(
        PLUGIN_NAME,
        inputs=[qkv_tensor, dw_weight, pw_weight],
        outputs=[plugin_output],
        name=f"{block_prefix}/EdgesegAggregationReluLinearAttention_TRT",
        plugin_version=PLUGIN_VERSION,
        plugin_namespace=PLUGIN_NAMESPACE,
        dim=int(args.dim),
        eps=float(args.eps),
        qkv_c=int(args.qkv_c),
        output_c=int(args.output_c),
        height=int(args.height),
        width=int(args.width),
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
    init_shapes = initializer_shapes(model)

    remove_names: Set[str] = set()
    removed_tensors: Set[str] = set()
    insertions: Dict[int, Any] = {}
    block_meta: List[Dict[str, Any]] = []

    for target in TARGET_BLOCKS:
        block_prefix = target["block_prefix"]
        initializer_prefix = target["initializer_prefix"]
        plugin_input = f"{block_prefix}/qkv/conv/Conv_output_0"
        plugin_output = f"{block_prefix}/Cast_1_output_0"
        dw_weight = f"{initializer_prefix}.aggreg.0.0.weight"
        pw_weight = f"{initializer_prefix}.aggreg.0.1.weight"

        if plugin_input not in consumers:
            raise RuntimeError(f"P1b plugin input has no consumer: {plugin_input}")
        if plugin_output not in producers:
            raise RuntimeError(f"P1b plugin output producer not found: {plugin_output}")
        for weight_name in (dw_weight, pw_weight):
            if weight_name not in init_shapes:
                raise RuntimeError(f"P1b aggregation weight initializer not found: {weight_name}")

        block_remove = collect_reverse_subgraph(producers, plugin_output, plugin_input)
        if not block_remove:
            raise RuntimeError(f"No removable P1b subgraph found for {block_prefix}")

        block_indices = [i for i, node in enumerate(original_nodes) if node.name in block_remove]
        insert_index = min(block_indices)
        plugin_node = make_plugin_node(onnx, block_prefix, initializer_prefix, args)
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
                "aggregation_weights": [
                    {"name": dw_weight, "shape": init_shapes[dw_weight]},
                    {"name": pw_weight, "shape": init_shapes[pw_weight]},
                ],
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
        preserved_tensors={target["block_prefix"] + "/Cast_1_output_0" for target in TARGET_BLOCKS},
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
            "note": "ONNX checker may not know TensorRT custom Plugin ops; parser build is the authoritative check.",
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
        "purpose": "phase3_p1b_aggregation_attention_plugin_onnx_integration",
        "scope": "real_efficientvit_graph_p1b_aggregation_plus_cat_plus_relu_linear_att_skeleton",
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
            "qkv_c": int(args.qkv_c),
            "output_c": int(args.output_c),
            "height": int(args.height),
            "width": int(args.width),
        },
        "graph": {
            "original_node_count": original_node_count,
            "patched_node_count": len(model.graph.node),
            "plugin_node_count": plugin_node_count,
            "target_block_count": len(TARGET_BLOCKS),
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
            "This is P1b aggregation + cat + relu_linear_att graph surgery.",
            "It preserves qkv, proj, and residual add outside the Plugin.",
            "Aggregation weights are passed as Plugin initializer inputs.",
            "Numerical correctness and latency are later responsibilities.",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }
    save_json(args.metadata.expanduser().resolve(), payload)
    return payload


def save_failure_metadata(args: argparse.Namespace, error: Exception) -> None:
    payload = {
        "status": "failed",
        "purpose": "phase3_p1b_aggregation_attention_plugin_onnx_integration",
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
        "P1b Plugin ONNX integration complete: "
        f"output={payload['output_onnx']['path']} "
        f"metadata={args.metadata} "
        f"plugin_nodes={payload['graph']['plugin_node_count']} "
        f"nodes={payload['graph']['original_node_count']}->{payload['graph']['patched_node_count']}"
    )


if __name__ == "__main__":
    main()
