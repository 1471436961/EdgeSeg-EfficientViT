"""Build a real EfficientViT TensorRT engine from the P1b Plugin-patched ONNX.

The current P1b Plugin contains a correctness-validated first CUDA math path.
This script only proves parser/build integration for the full EfficientViT graph;
end-to-end correctness and latency are validated by benchmark scripts.
"""

from __future__ import annotations

import argparse
import ctypes
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_NAME = "build_p1b_plugin_engine.py"
PLUGIN_NAME = "EdgesegAggregationReluLinearAttention_TRT"
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "edgeseg"
DEFAULT_ONNX = Path("phase3/results/onnx/efficientvit_seg_b0_cityscapes_1024x2048_p1b_aggregation_attention_plugin.onnx")
DEFAULT_ENGINE = Path(
    "phase3/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_p1b_aggregation_attention_plugin_fp32.engine"
)
DEFAULT_METADATA = Path("phase3/results/metrics/p1b_aggregation_attention_plugin_engine_build.json")
DEFAULT_WORKSPACE_MIB = 1024


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
PHASE2_SCRIPTS = ROOT / "phase2" / "scripts"
if str(PHASE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PHASE2_SCRIPTS))

from _common import resolve_script_version, save_json, sha256_of_file, version_of  # noqa: E402
from _trt_runtime import DEFAULT_TRT_ROOT, prepare_runtime_paths  # noqa: E402
from build_plugin_toy_engine import (  # noqa: E402
    add_phase3_runtime_dirs,
    creator_names,
    default_plugin_dll,
    set_workspace_limit,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an EfficientViT TensorRT engine from P1b Plugin-patched ONNX.")
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="P1b Plugin-patched ONNX path.")
    p.add_argument("--engine", type=Path, default=DEFAULT_ENGINE, help="Output P1b Plugin TensorRT engine path.")
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="Build metadata JSON path.")
    p.add_argument("--plugin-dll", type=Path, default=default_plugin_dll(), help="Plugin DLL path.")
    p.add_argument("--trt-root", type=Path, default=DEFAULT_TRT_ROOT, help="TensorRT zip root directory.")
    p.add_argument("--workspace-mib", type=int, default=DEFAULT_WORKSPACE_MIB)
    p.add_argument("--verbose", action="store_true", help="Use verbose TensorRT logger.")
    return p.parse_args()


def input_shape_to_list(shape) -> List[int]:
    return [int(shape[i]) for i in range(len(shape))]


def collect_network_io(network) -> Dict[str, Any]:
    inputs = []
    outputs = []
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        inputs.append({"name": tensor.name, "shape": input_shape_to_list(tensor.shape), "dtype": str(tensor.dtype)})
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        outputs.append({"name": tensor.name, "shape": input_shape_to_list(tensor.shape), "dtype": str(tensor.dtype)})
    return {"inputs": inputs, "outputs": outputs, "num_layers": int(network.num_layers)}


def build_engine(args: argparse.Namespace) -> Dict[str, Any]:
    runtime_meta = prepare_runtime_paths(args.trt_root.expanduser().resolve())
    add_phase3_runtime_dirs(runtime_meta)

    try:
        import tensorrt as trt
    except Exception as exc:
        raise RuntimeError(f"failed to import TensorRT after DLL path setup: {exc}") from exc

    plugin_dll = args.plugin_dll.expanduser().resolve()
    if not plugin_dll.is_file():
        raise FileNotFoundError(f"Plugin DLL not found: {plugin_dll}")
    dll_handle = ctypes.CDLL(str(plugin_dll))

    onnx_path = args.onnx.expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"P1b Plugin ONNX file not found: {onnx_path}")

    severity = trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO
    logger = trt.Logger(severity)
    if hasattr(trt, "init_libnvinfer_plugins"):
        trt.init_libnvinfer_plugins(logger, "")

    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator(PLUGIN_NAME, PLUGIN_VERSION, PLUGIN_NAMESPACE)
    registered_creators = creator_names(registry)
    if creator is None:
        raise RuntimeError(
            f"Plugin creator not found: name={PLUGIN_NAME}, version={PLUGIN_VERSION}, namespace={PLUGIN_NAMESPACE}"
        )

    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    parse_ok = parser.parse(onnx_path.read_bytes())
    parser_errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not parse_ok:
        raise RuntimeError("TensorRT ONNX parser failed:\n" + "\n".join(parser_errors))

    config = builder.create_builder_config()
    workspace_bytes = int(args.workspace_mib) * 1024 * 1024
    workspace_method = set_workspace_limit(config, trt, workspace_bytes)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT builder returned None serialized P1b Plugin engine")

    engine_path = args.engine.expanduser().resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))

    metadata = {
        "status": "ok",
        "purpose": "phase3_p1b_aggregation_attention_plugin_engine_build",
        "scope": "real_efficientvit_graph_p1b_aggregation_plus_cat_plus_relu_linear_att",
        "plugin": {
            "name": PLUGIN_NAME,
            "version": PLUGIN_VERSION,
            "namespace": PLUGIN_NAMESPACE,
            "dll_path": str(plugin_dll),
            "dll_sha256": sha256_of_file(plugin_dll),
            "creator_found": True,
            "registered_creator_count": len(registered_creators),
        },
        "onnx": {
            "onnx_path": str(onnx_path),
            "onnx_sha256": sha256_of_file(onnx_path),
            "onnx_size_bytes": onnx_path.stat().st_size,
        },
        "engine": {
            "engine_path": str(engine_path),
            "engine_sha256": sha256_of_file(engine_path),
            "engine_size_bytes": engine_path.stat().st_size,
        },
        "tensorrt": {
            "version": trt.__version__,
            "workspace_mib": int(args.workspace_mib),
            "workspace_bytes": workspace_bytes,
            "workspace_method": workspace_method,
            "network_definition": "explicit_batch",
            "parser_errors": parser_errors,
            "precision": "fp32",
        },
        "network": collect_network_io(network),
        "runtime_paths": runtime_meta,
        "versions": {
            "python": platform.python_version(),
            "tensorrt": version_of("tensorrt"),
            "numpy": version_of("numpy"),
        },
        "notes": [
            "This validates P1b Plugin-patched ONNX parser/build integration for the full EfficientViT graph.",
            "The current P1b Plugin contains a first CUDA math path that passed block-level toy/plugin validation.",
            "This build metadata alone does not prove end-to-end correctness or latency; use benchmark_plugin_engine.py.",
            "Engine is specific to the current TensorRT version and GPU.",
        ],
        "_dll_handle_alive": bool(dll_handle),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }
    save_json(args.metadata.expanduser().resolve(), metadata)
    return metadata


def save_failure_metadata(args: argparse.Namespace, error: Exception) -> None:
    payload = {
        "status": "failed",
        "purpose": "phase3_p1b_aggregation_attention_plugin_engine_build",
        "scope": "real_efficientvit_graph_p1b_aggregation_plus_cat_plus_relu_linear_att",
        "plugin_dll": str(args.plugin_dll.expanduser().resolve()),
        "onnx": str(args.onnx.expanduser().resolve()),
        "engine": str(args.engine.expanduser().resolve()),
        "error_type": type(error).__name__,
        "error": str(error),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }
    save_json(args.metadata.expanduser().resolve(), payload)


def main() -> None:
    args = parse_args()
    try:
        metadata = build_engine(args)
    except Exception as exc:
        save_failure_metadata(args, exc)
        raise

    print(
        "P1b Plugin TensorRT engine build complete: "
        f"engine={metadata['engine']['engine_path']} "
        f"metadata={args.metadata} "
        f"layers={metadata['network']['num_layers']}"
    )


if __name__ == "__main__":
    main()
