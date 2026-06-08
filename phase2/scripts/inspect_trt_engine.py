#!/usr/bin/env python3
"""Inspect TensorRT engine structure and map it back to ONNX node names.

This script is structural evidence only. Runtime attribution remains owned by
analyze_trt_nsys_attribution.py and Nsight SQLite correlationId joins.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import site
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPT_NAME = "inspect_trt_engine.py"
DEFAULT_TRT_ROOT = Path(r"E:\NVIDIA\TensorRT-8.6.1.6")
DEFAULT_ENGINE = Path("phase2/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_fp32.engine")
DEFAULT_ONNX = Path("phase2/results/onnx/efficientvit_seg_b0_cityscapes_1024x2048.onnx")
DEFAULT_OUT_JSON = Path("phase2/results/metrics/trt_engine_inspection_summary.json")
DEFAULT_OUT_MD = Path("phase2/results/metrics/trt_engine_inspection_summary.md")
_DLL_DIRECTORY_HANDLES: List[object] = []


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect TensorRT engine and ONNX node mapping.")
    p.add_argument("--engine", type=Path, default=DEFAULT_ENGINE, help="Input TensorRT engine.")
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="Input ONNX graph.")
    p.add_argument("--trt-root", type=Path, default=DEFAULT_TRT_ROOT, help="TensorRT zip root directory.")
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON, help="JSON summary output.")
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD, help="Markdown summary output.")
    return p.parse_args()


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def version_of(package: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_script_version() -> str:
    root = repo_root()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
        rel = Path(__file__).resolve().relative_to(root)
        diff = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(rel)],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        suffix = "-dirty" if diff.returncode == 1 else ""
        return f"{SCRIPT_NAME}@{commit}{suffix}"
    except Exception:
        return f"{SCRIPT_NAME}@unknown"


def candidate_runtime_dirs(trt_root: Path) -> List[Path]:
    dirs = [trt_root / "lib", trt_root / "bin"]
    for site_dir in site.getsitepackages():
        base = Path(site_dir) / "nvidia"
        dirs.extend(
            [
                base / "cudnn" / "bin",
                base / "cublas" / "bin",
                base / "cuda_nvrtc" / "bin",
            ]
        )
    return dirs


def prepare_runtime_paths(trt_root: Path) -> Dict[str, Any]:
    added: List[str] = []
    missing: List[str] = []
    for path in candidate_runtime_dirs(trt_root):
        if path.is_dir():
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))
            os.environ["PATH"] = f"{path}{os.pathsep}{os.environ.get('PATH', '')}"
            added.append(str(path))
        else:
            missing.append(str(path))
    return {
        "trt_root": str(trt_root),
        "dll_dirs_added": added,
        "candidate_dirs_missing": missing,
    }


def group_name(name: str) -> str:
    if "/backbone/input_stem" in name:
        return "stem"
    match = re.search(r"/backbone/stages\.(\d+)", name)
    if match:
        return f"stage{match.group(1)}"
    if "/head/" in name:
        return "head"
    if name.startswith("(Unnamed Layer") or name.startswith("Constant"):
        return "constant/unnamed"
    return "other"


def extract_onnx_paths(layer_name: str) -> List[str]:
    paths = re.findall(r"/[A-Za-z0-9_./]+", layer_name)
    out: List[str] = []
    for path in paths:
        cleaned = path.rstrip(".,)")
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def layer_kind(layer_name: str) -> str:
    if layer_name.startswith("(Unnamed Layer"):
        return "constant"
    if layer_name.startswith("PWN("):
        return "pointwise_fusion"
    if " + " in layer_name:
        return "explicit_fusion"
    if "Resize" in layer_name:
        return "resize"
    for token in ["MatMul", "Conv", "Concat", "Slice", "Shuffle", "Reduce", "Softmax"]:
        if token in layer_name:
            return token.lower()
    return "single_or_other"


def load_onnx_nodes(onnx_path: Path) -> List[Dict[str, Any]]:
    import onnx

    model = onnx.load(str(onnx_path))
    nodes: List[Dict[str, Any]] = []
    for index, node in enumerate(model.graph.node):
        name = node.name or f"<unnamed_{index}>"
        nodes.append(
            {
                "index": index,
                "name": name,
                "op_type": node.op_type,
                "group": group_name(name),
                "input_count": len(node.input),
                "output_count": len(node.output),
            }
        )
    return nodes


def load_engine_layers(engine_path: Path, trt_root: Path) -> Dict[str, Any]:
    runtime_meta = prepare_runtime_paths(trt_root.expanduser().resolve())

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")

    inspector = engine.create_engine_inspector()
    engine_info_raw = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    try:
        engine_info = json.loads(engine_info_raw)
    except json.JSONDecodeError:
        engine_info = {"raw": engine_info_raw}

    names: List[str] = []
    if isinstance(engine_info, dict) and isinstance(engine_info.get("Layers"), list):
        names = [str(name) for name in engine_info["Layers"]]
    else:
        for index in range(int(engine.num_layers)):
            raw = inspector.get_layer_information(index, trt.LayerInformationFormat.JSON)
            try:
                parsed = json.loads(raw)
                names.append(str(parsed.get("Name", parsed)))
            except Exception:
                names.append(str(raw).strip().strip('"'))

    layers = []
    detail_is_name_only = True
    for index, name in enumerate(names):
        raw = inspector.get_layer_information(index, trt.LayerInformationFormat.JSON)
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = raw
        if isinstance(parsed, dict):
            detail_is_name_only = False
        layers.append(
            {
                "index": index,
                "name": name,
                "group": group_name(name),
                "kind": layer_kind(name),
                "onnx_paths": extract_onnx_paths(name),
                "raw_detail": parsed,
            }
        )

    binding_meta = []
    if hasattr(engine, "num_io_tensors"):
        for index in range(int(engine.num_io_tensors)):
            name = engine.get_tensor_name(index)
            mode = str(engine.get_tensor_mode(name))
            binding_meta.append(
                {
                    "index": index,
                    "name": name,
                    "is_input": "INPUT" in mode,
                    "mode": mode,
                    "shape": [int(dim) for dim in engine.get_tensor_shape(name)],
                    "dtype": str(engine.get_tensor_dtype(name)),
                }
            )
    else:
        for index in range(int(engine.num_bindings)):
            binding_meta.append(
                {
                    "index": index,
                    "name": engine.get_binding_name(index),
                    "is_input": bool(engine.binding_is_input(index)),
                    "shape": [int(dim) for dim in engine.get_binding_shape(index)],
                    "dtype": str(engine.get_binding_dtype(index)),
                }
            )

    return {
        "runtime_paths": runtime_meta,
        "tensorrt": {
            "version": trt.__version__,
            "num_layers": int(engine.num_layers),
            "num_bindings": int(engine.num_bindings),
            "inspector_detail": "layer_names_only" if detail_is_name_only else "detailed",
        },
        "bindings": binding_meta,
        "engine_info": engine_info,
        "layers": layers,
    }


def summarize(onnx_nodes: List[Dict[str, Any]], engine_layers: List[Dict[str, Any]]) -> Dict[str, Any]:
    onnx_by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node in onnx_nodes:
        onnx_by_group[node["group"]].append(node)

    trt_by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for layer in engine_layers:
        trt_by_group[layer["group"]].append(layer)

    groups = []
    for group in sorted(set(onnx_by_group) | set(trt_by_group)):
        nodes = onnx_by_group.get(group, [])
        layers = trt_by_group.get(group, [])
        onnx_count = len(nodes)
        trt_count = len(layers)
        fused_count = sum(1 for layer in layers if layer["kind"] in {"pointwise_fusion", "explicit_fusion"})
        pointwise_count = sum(1 for layer in layers if layer["kind"] == "pointwise_fusion")
        explicit_count = sum(1 for layer in layers if layer["kind"] == "explicit_fusion")
        groups.append(
            {
                "group": group,
                "onnx_nodes": onnx_count,
                "trt_layers": trt_count,
                "layer_count_delta": trt_count - onnx_count,
                "layer_reduction_pct": 100.0 * (onnx_count - trt_count) / onnx_count if onnx_count else 0.0,
                "trt_fused_layers": fused_count,
                "trt_pointwise_fusion_layers": pointwise_count,
                "trt_explicit_fusion_layers": explicit_count,
                "onnx_op_types": Counter(node["op_type"] for node in nodes).most_common(12),
                "trt_layer_kinds": Counter(layer["kind"] for layer in layers).most_common(12),
            }
        )

    return {
        "onnx_node_count": len(onnx_nodes),
        "trt_layer_count": len(engine_layers),
        "overall_layer_reduction_pct": 100.0 * (len(onnx_nodes) - len(engine_layers)) / len(onnx_nodes)
        if onnx_nodes
        else 0.0,
        "groups": sorted(groups, key=lambda row: row["trt_layers"], reverse=True),
        "onnx_op_types": Counter(node["op_type"] for node in onnx_nodes).most_common(30),
        "trt_layer_kinds": Counter(layer["kind"] for layer in engine_layers).most_common(30),
    }


def interesting_stage2_layers(layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for layer in layers:
        name = layer["name"]
        if "/backbone/stages.2/" not in name:
            continue
        if "context_module" not in name:
            continue
        out.append(
            {
                "index": layer["index"],
                "name": name,
                "kind": layer["kind"],
                "onnx_paths": layer["onnx_paths"],
            }
        )
    return out


def render_markdown(payload: Dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# TensorRT Engine Inspection Summary",
        "",
        f"- Engine: `{payload['engine']['engine_path']}`",
        f"- ONNX: `{payload['onnx']['onnx_path']}`",
        f"- TensorRT: `{payload['tensorrt']['version']}`",
        f"- EngineInspector detail: `{payload['tensorrt']['inspector_detail']}`",
        f"- ONNX node count: {summary['onnx_node_count']}",
        f"- TensorRT engine layer count: {summary['trt_layer_count']}",
        f"- Overall layer-count reduction: {summary['overall_layer_reduction_pct']:.2f}%",
        "",
        "This file is structural evidence. It does not contain runtime timing; use `trt_nsys_attribution_summary.md` for GPU kernel time.",
        "",
        "## Group Mapping Summary",
        "",
        "| Group | ONNX nodes | TRT layers | Layer reduction | TRT fused layers | PWN layers | Explicit `+` fusion layers |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["groups"]:
        lines.append(
            f"| `{row['group']}` | {row['onnx_nodes']} | {row['trt_layers']} | "
            f"{row['layer_reduction_pct']:.2f}% | {row['trt_fused_layers']} | "
            f"{row['trt_pointwise_fusion_layers']} | {row['trt_explicit_fusion_layers']} |"
        )

    lines.extend(
        [
            "",
            "## ONNX Op Type Summary",
            "",
            "| Op type | Count |",
            "|---|---:|",
        ]
    )
    for op_type, count in summary["onnx_op_types"][:20]:
        lines.append(f"| `{op_type}` | {count} |")

    lines.extend(
        [
            "",
            "## TensorRT Layer Kind Summary",
            "",
            "| Layer kind | Count |",
            "|---|---:|",
        ]
    )
    for kind, count in summary["trt_layer_kinds"][:20]:
        lines.append(f"| `{kind}` | {count} |")

    lines.extend(
        [
            "",
            "## Stage2 Context Engine Layers",
            "",
            "| Index | Kind | TensorRT layer name |",
            "|---:|---|---|",
        ]
    )
    for row in payload["stage2_context_layers"][:80]:
        name = row["name"].replace("|", "\\|")
        lines.append(f"| {row['index']} | `{row['kind']}` | `{name}` |")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `PWN(...)` layers indicate TensorRT pointwise/activation fusion.",
            "- Layer names containing ` + ` indicate TensorRT fused multiple ONNX-named operations into one engine layer.",
            "- The current FP32 engine exposes layer names only, not detailed tactic metadata. For tactic-level evidence, rebuild with detailed profiling verbosity or capture verbose builder logs.",
            "- EngineInspector structure is auxiliary evidence; Nsight SQLite attribution remains the source of runtime GPU time.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    engine_path = args.engine.expanduser().resolve()
    onnx_path = args.onnx.expanduser().resolve()
    if not engine_path.is_file():
        raise FileNotFoundError(f"engine not found: {engine_path}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    onnx_nodes = load_onnx_nodes(onnx_path)
    engine_payload = load_engine_layers(engine_path, args.trt_root)
    engine_layers = engine_payload["layers"]
    summary = summarize(onnx_nodes, engine_layers)

    payload = {
        "status": "ok",
        "engine": {
            "engine_path": str(engine_path),
            "engine_sha256": sha256_of_file(engine_path),
            "engine_size_bytes": engine_path.stat().st_size,
        },
        "onnx": {
            "onnx_path": str(onnx_path),
            "onnx_sha256": sha256_of_file(onnx_path),
            "onnx_size_bytes": onnx_path.stat().st_size,
        },
        "tensorrt": engine_payload["tensorrt"],
        "bindings": engine_payload["bindings"],
        "runtime_paths": engine_payload["runtime_paths"],
        "summary": summary,
        "engine_layers": engine_layers,
        "onnx_nodes": onnx_nodes,
        "stage2_context_layers": interesting_stage2_layers(engine_layers),
        "versions": {
            "python": platform.python_version(),
            "onnx": version_of("onnx"),
            "tensorrt": version_of("tensorrt"),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"TensorRT engine inspection written: {args.out_md}")


if __name__ == "__main__":
    main()
