"""Build a toy TensorRT engine for the P1b aggregation + attention Plugin.

This script intentionally exercises the TensorRT ONNX parser path instead of
the TensorRT Network API path. The question it answers is narrow:

Can TensorRT 8.6.1 parse a custom Plugin node whose inputs are:

1. A runtime qkv tensor.
2. A depthwise aggregation weight initializer.
3. A grouped pointwise aggregation weight initializer.

The P1b Plugin implementation in this step is a skeleton that zero-fills its
output. Numerical correctness and performance are later steps.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


SCRIPT_NAME = "build_p1b_plugin_toy_engine.py"
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
from _trt_runtime import DEFAULT_TRT_ROOT, prepare_runtime_paths  # noqa: E402
from build_plugin_toy_engine import (  # noqa: E402
    add_phase3_runtime_dirs,
    creator_names,
    default_plugin_dll,
    dims_to_list,
    set_workspace_limit,
)


def default_toy_onnx() -> Path:
    return Path("phase3/results/onnx/p1b_aggregation_attention_toy.onnx")


def default_engine_path() -> Path:
    return Path("phase3/results/engines/p1b_aggregation_attention_toy_fp32.engine")


def default_metadata_path() -> Path:
    return Path("phase3/results/metrics/p1b_aggregation_attention_toy_build.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a parser toy engine for the P1b Plugin skeleton.")
    parser.add_argument("--plugin-dll", type=Path, default=default_plugin_dll(), help="Plugin DLL path.")
    parser.add_argument("--toy-onnx", type=Path, default=default_toy_onnx(), help="Generated toy ONNX path.")
    parser.add_argument("--engine", type=Path, default=default_engine_path(), help="Output toy engine path.")
    parser.add_argument("--metadata", type=Path, default=default_metadata_path(), help="Output metadata JSON path.")
    parser.add_argument("--trt-root", type=Path, default=DEFAULT_TRT_ROOT, help="TensorRT zip root directory.")
    parser.add_argument("--workspace-mib", type=int, default=256, help="Workspace limit in MiB.")
    parser.add_argument("--verbose", action="store_true", help="Use verbose TensorRT logger.")
    return parser.parse_args()


def make_weight(shape: tuple[int, ...], scale: float) -> np.ndarray:
    count = int(np.prod(shape))
    values = (np.arange(count, dtype=np.float32) % 127) / 127.0
    return (values.reshape(shape) * scale).astype(np.float32)


def build_toy_onnx(path: Path) -> Dict[str, Any]:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    qkv = helper.make_tensor_value_info("qkv", TensorProto.FLOAT, [1, 192, 64, 128])
    output = helper.make_tensor_value_info("attention_out", TensorProto.FLOAT, [1, 128, 64, 128])

    dw_weight = make_weight((192, 1, 5, 5), 0.01)
    pw_weight = make_weight((192, 16, 1, 1), 0.01)
    dw_init = numpy_helper.from_array(dw_weight, name="aggregation_depthwise_weight")
    pw_init = numpy_helper.from_array(pw_weight, name="aggregation_pointwise_weight")

    plugin_node = helper.make_node(
        PLUGIN_NAME,
        inputs=["qkv", "aggregation_depthwise_weight", "aggregation_pointwise_weight"],
        outputs=["attention_out"],
        name="p1b_aggregation_attention_toy_plugin",
        plugin_version=PLUGIN_VERSION,
        plugin_namespace=PLUGIN_NAMESPACE,
        dim=16,
        eps=1.0e-15,
        qkv_c=192,
        output_c=128,
        height=64,
        width=128,
    )

    graph = helper.make_graph(
        [plugin_node],
        "p1b_aggregation_attention_toy_graph",
        [qkv],
        [output],
        initializer=[dw_init, pw_init],
    )
    model = helper.make_model(
        graph,
        producer_name="edgeseg_phase3_p1b_toy",
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    onnx.save(model, str(path))

    checker: Dict[str, Any]
    try:
        onnx.checker.check_model(model)
        checker = {"status": "ok", "error": None}
    except Exception as exc:
        checker = {
            "status": "warning",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "note": "ONNX checker may not know TensorRT custom Plugin ops; TensorRT parser build is authoritative.",
        }

    return {
        "path": str(path),
        "sha256": sha256_of_file(path),
        "size_bytes": path.stat().st_size,
        "checker": checker,
        "node": {
            "op_type": PLUGIN_NAME,
            "inputs": list(plugin_node.input),
            "outputs": list(plugin_node.output),
        },
        "initializers": [
            {"name": "aggregation_depthwise_weight", "shape": list(dw_weight.shape), "dtype": str(dw_weight.dtype)},
            {"name": "aggregation_pointwise_weight", "shape": list(pw_weight.shape), "dtype": str(pw_weight.dtype)},
        ],
        "versions": {"onnx": onnx.__version__},
    }


def collect_network_io(network) -> Dict[str, Any]:
    inputs = []
    outputs = []
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        inputs.append({"name": tensor.name, "shape": dims_to_list(tensor.shape), "dtype": str(tensor.dtype)})
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        outputs.append({"name": tensor.name, "shape": dims_to_list(tensor.shape), "dtype": str(tensor.dtype)})
    return {"inputs": inputs, "outputs": outputs, "num_layers": int(network.num_layers)}


def build_toy_engine(args: argparse.Namespace) -> Dict[str, Any]:
    runtime_meta = prepare_runtime_paths(args.trt_root.expanduser().resolve())
    add_phase3_runtime_dirs(runtime_meta)
    toy_onnx_meta = build_toy_onnx(args.toy_onnx)

    try:
        import tensorrt as trt
    except Exception as exc:
        raise RuntimeError(f"failed to import TensorRT after DLL path setup: {exc}") from exc

    plugin_dll = args.plugin_dll.expanduser().resolve()
    if not plugin_dll.is_file():
        raise FileNotFoundError(f"Plugin DLL not found: {plugin_dll}")

    dll_handle = ctypes.CDLL(str(plugin_dll))
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

    toy_onnx_path = Path(toy_onnx_meta["path"])
    parse_ok = parser.parse(toy_onnx_path.read_bytes())
    parser_errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not parse_ok:
        raise RuntimeError("TensorRT ONNX parser failed:\n" + "\n".join(parser_errors))

    config = builder.create_builder_config()
    workspace_bytes = int(args.workspace_mib) * 1024 * 1024
    workspace_method = set_workspace_limit(config, trt, workspace_bytes)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT builder returned None serialized toy engine")

    engine_path = args.engine.expanduser().resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))

    metadata = {
        "status": "ok",
        "purpose": "phase3_p1b_parser_toy_engine_build",
        "scope": "p1b_aggregation_plus_cat_plus_relu_linear_att_skeleton",
        "plugin": {
            "name": PLUGIN_NAME,
            "version": PLUGIN_VERSION,
            "namespace": PLUGIN_NAMESPACE,
            "dll_path": str(plugin_dll),
            "dll_sha256": sha256_of_file(plugin_dll),
            "creator_found": True,
            "registered_creator_count": len(registered_creators),
        },
        "toy_onnx": toy_onnx_meta,
        "engine": {
            "engine_path": str(engine_path),
            "engine_sha256": sha256_of_file(engine_path),
            "engine_size_bytes": engine_path.stat().st_size,
        },
        "tensorrt": {
            "version": trt.__version__,
            "workspace_mib": args.workspace_mib,
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
            "onnx": version_of("onnx"),
        },
        "notes": [
            "This validates P1b Plugin Creator registration and TensorRT ONNX parser/build only.",
            "The P1b Plugin skeleton zero-fills its output; this is not a correctness or latency artifact.",
            "The key parser question is whether initializer weights can be accepted as Plugin inputs.",
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
        "purpose": "phase3_p1b_parser_toy_engine_build",
        "plugin_dll": str(args.plugin_dll.expanduser().resolve()),
        "toy_onnx": str(args.toy_onnx.expanduser().resolve()),
        "engine_path": str(args.engine.expanduser().resolve()),
        "trt_root": str(args.trt_root.expanduser().resolve()),
        "error_type": type(error).__name__,
        "error": str(error),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }
    save_json(args.metadata.expanduser().resolve(), payload)


def main() -> None:
    args = parse_args()
    try:
        metadata = build_toy_engine(args)
    except Exception as exc:
        save_failure_metadata(args, exc)
        raise

    print(
        "P1b parser toy engine build complete: "
        f"engine={metadata['engine']['engine_path']} "
        f"plugin={metadata['plugin']['dll_path']} "
        f"layers={metadata['network']['num_layers']}"
    )


if __name__ == "__main__":
    main()
