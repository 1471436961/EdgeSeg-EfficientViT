"""Build a minimal TensorRT engine containing the Phase 3 Plugin skeleton.

This script validates only the Plugin integration path:

1. TensorRT/CUDA DLL paths are visible.
2. The Plugin DLL can be loaded.
3. The Plugin creator is registered under the expected namespace.
4. A toy explicit-batch network can be built with the Plugin layer.

It does not validate real LiteMLA math. The Step 4 skeleton Plugin currently
fills its output with zero; Step 5 will replace that with the real kernel.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


SCRIPT_NAME = "build_plugin_toy_engine.py"
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
from _trt_runtime import DEFAULT_TRT_ROOT, prepare_runtime_paths  # noqa: E402

_DLL_DIRECTORY_HANDLES: List[object] = []


def default_plugin_dll() -> Path:
    return Path("phase3/plugin/build/edgeseg_relu_linear_attention_plugin.dll")


def default_engine_path() -> Path:
    return Path("phase3/results/engines/relu_linear_attention_toy_fp32.engine")


def default_metadata_path() -> Path:
    return Path("phase3/results/metrics/relu_linear_attention_toy_build.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a toy TensorRT engine with the Phase 3 Plugin skeleton.")
    parser.add_argument("--plugin-dll", type=Path, default=default_plugin_dll(), help="Plugin DLL path.")
    parser.add_argument("--engine", type=Path, default=default_engine_path(), help="Output toy engine path.")
    parser.add_argument("--metadata", type=Path, default=default_metadata_path(), help="Output metadata JSON path.")
    parser.add_argument("--trt-root", type=Path, default=DEFAULT_TRT_ROOT, help="TensorRT zip root directory.")
    parser.add_argument("--workspace-mib", type=int, default=256, help="Workspace limit in MiB.")
    parser.add_argument("--verbose", action="store_true", help="Use verbose TensorRT logger.")
    return parser.parse_args()


def set_workspace_limit(config, trt, workspace_bytes: int) -> str:
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        return "set_memory_pool_limit"
    config.max_workspace_size = workspace_bytes
    return "max_workspace_size"


def add_phase3_runtime_dirs(runtime_meta: Dict[str, Any]) -> None:
    """Add CUDA Toolkit DLL dirs needed by the Plugin DLL itself."""
    cuda_path = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"))
    candidates = [cuda_path / "bin"]
    added: List[str] = []
    missing: List[str] = []
    for path in candidates:
        if path.is_dir():
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))
            os.environ["PATH"] = f"{path}{os.pathsep}{os.environ.get('PATH', '')}"
            added.append(str(path))
        else:
            missing.append(str(path))
    runtime_meta["phase3_extra_dll_dirs_added"] = added
    runtime_meta["phase3_extra_dll_dirs_missing"] = missing


def dims_to_list(dims) -> List[int]:
    return [int(dims[i]) for i in range(len(dims))]


def creator_names(registry) -> List[Dict[str, str]]:
    creators = getattr(registry, "plugin_creator_list", [])
    out: List[Dict[str, str]] = []
    for creator in creators:
        out.append(
            {
                "name": str(creator.name),
                "version": str(creator.plugin_version),
                "namespace": str(creator.plugin_namespace),
            }
        )
    return out


def make_plugin_fields(trt):
    fields = [
        trt.PluginField("dim", np.array([16], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("eps", np.array([1.0e-15], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("input_c", np.array([384], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("height", np.array([64], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("width", np.array([128], dtype=np.int32), trt.PluginFieldType.INT32),
    ]
    return trt.PluginFieldCollection(fields)


def build_toy_engine(args: argparse.Namespace) -> Dict[str, Any]:
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

    input_tensor = network.add_input("qkv_cat", trt.float32, (1, 384, 64, 128))
    if input_tensor is None:
        raise RuntimeError("failed to add toy input tensor")

    plugin = creator.create_plugin("relu_linear_attention_toy", make_plugin_fields(trt))
    if plugin is None:
        raise RuntimeError("creator.create_plugin returned None")

    layer = network.add_plugin_v2([input_tensor], plugin)
    if layer is None:
        raise RuntimeError("network.add_plugin_v2 returned None")
    layer.name = "relu_linear_attention_toy_plugin"
    output_tensor = layer.get_output(0)
    output_tensor.name = "attention_out"
    network.mark_output(output_tensor)

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
        "purpose": "phase3_step4_plugin_skeleton_toy_engine",
        "plugin": {
            "name": PLUGIN_NAME,
            "version": PLUGIN_VERSION,
            "namespace": PLUGIN_NAMESPACE,
            "dll_path": str(plugin_dll),
            "dll_sha256": sha256_of_file(plugin_dll),
            "creator_found": True,
            "registered_creator_count": len(registered_creators),
        },
        "engine": {
            "engine_path": str(engine_path),
            "engine_sha256": sha256_of_file(engine_path),
            "engine_size_bytes": engine_path.stat().st_size,
        },
        "toy_network": {
            "input": {"name": input_tensor.name, "shape": dims_to_list(input_tensor.shape), "dtype": str(input_tensor.dtype)},
            "output": {"name": output_tensor.name, "shape": dims_to_list(output_tensor.shape), "dtype": str(output_tensor.dtype)},
            "num_layers": int(network.num_layers),
        },
        "tensorrt": {
            "version": trt.__version__,
            "workspace_mib": args.workspace_mib,
            "workspace_bytes": workspace_bytes,
            "workspace_method": workspace_method,
            "network_definition": "explicit_batch",
        },
        "runtime_paths": runtime_meta,
        "versions": {
            "python": platform.python_version(),
            "tensorrt": version_of("tensorrt"),
            "numpy": version_of("numpy"),
        },
        "notes": [
            "Step 4 validates Plugin registration and toy engine build only.",
            "The skeleton Plugin currently zero-fills its output; real LiteMLA math is Step 5.",
            "This toy engine is not a performance or correctness artifact.",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }

    metadata_path = args.metadata.expanduser().resolve()
    save_json(metadata_path, metadata)

    # Keep the handle alive until after the engine is built.
    metadata["_dll_handle_alive"] = bool(dll_handle)
    return metadata


def save_failure_metadata(args: argparse.Namespace, error: Exception) -> None:
    payload = {
        "status": "failed",
        "purpose": "phase3_step4_plugin_skeleton_toy_engine",
        "plugin_dll": str(args.plugin_dll.expanduser().resolve()),
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
        "Plugin toy engine build complete: "
        f"engine={metadata['engine']['engine_path']} "
        f"plugin={metadata['plugin']['dll_path']} "
        f"trt={metadata['tensorrt']['version']}"
    )


if __name__ == "__main__":
    main()
