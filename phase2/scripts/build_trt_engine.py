"""Build a fixed-shape TensorRT engine from the Phase 2 ONNX export.

This script owns only the ONNX -> TensorRT engine build step. Benchmarking and
output validation are intentionally left to a later script so parser/build
failures remain easy to diagnose.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import site
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPT_NAME = "build_trt_engine.py"
DEFAULT_TRT_ROOT = Path(r"E:\NVIDIA\TensorRT-8.6.1.6")
DEFAULT_ONNX = Path("phase2/results/onnx/efficientvit_seg_b0_cityscapes_1024x2048.onnx")
DEFAULT_ENGINE = Path("phase2/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_fp32.engine")
DEFAULT_METADATA = Path("phase2/results/metrics/trt_build_b0_cityscapes_1024x2048_fp32.json")
DEFAULT_WORKSPACE_MIB = 1024
_DLL_DIRECTORY_HANDLES: List[object] = []


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a TensorRT FP32 engine from ONNX.")
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="Input ONNX path.")
    p.add_argument("--engine", type=Path, default=DEFAULT_ENGINE, help="Output TensorRT engine path.")
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="Build metadata JSON path.")
    p.add_argument("--trt-root", type=Path, default=DEFAULT_TRT_ROOT, help="TensorRT zip root directory.")
    p.add_argument("--workspace-mib", type=int, default=DEFAULT_WORKSPACE_MIB, help="Workspace limit in MiB.")
    p.add_argument("--precision", choices=["fp32"], default="fp32", help="First build version supports FP32 only.")
    p.add_argument("--verbose", action="store_true", help="Use verbose TensorRT logger.")
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
    dirs: List[Path] = [
        trt_root / "lib",
        trt_root / "bin",
    ]
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
    """Make TensorRT/cuDNN/cuBLAS DLLs visible to this Python process."""
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


def set_workspace_limit(config, trt, workspace_bytes: int) -> str:
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        return "set_memory_pool_limit"
    config.max_workspace_size = workspace_bytes
    return "max_workspace_size"


def build_engine(args: argparse.Namespace) -> Dict[str, Any]:
    runtime_meta = prepare_runtime_paths(args.trt_root.expanduser().resolve())

    try:
        import tensorrt as trt
    except Exception as exc:
        raise RuntimeError(f"failed to import TensorRT after DLL path setup: {exc}") from exc

    onnx_path = args.onnx.expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    severity = trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = onnx_path.read_bytes()
    parse_ok = parser.parse(onnx_bytes)
    parser_errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not parse_ok:
        raise RuntimeError("TensorRT ONNX parser failed:\n" + "\n".join(parser_errors))

    config = builder.create_builder_config()
    workspace_bytes = int(args.workspace_mib) * 1024 * 1024
    workspace_method = set_workspace_limit(config, trt, workspace_bytes)

    if args.precision != "fp32":
        raise ValueError("only fp32 is supported in the first TensorRT build script")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT builder returned None serialized engine")

    engine_path = args.engine.expanduser().resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))

    metadata_path = args.metadata.expanduser().resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    build_meta = {
        "status": "ok",
        "precision": args.precision,
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
            "builder_created": True,
            "platform_has_fast_fp16": bool(builder.platform_has_fast_fp16),
            "platform_has_fast_int8": bool(builder.platform_has_fast_int8),
            "workspace_mib": args.workspace_mib,
            "workspace_bytes": workspace_bytes,
            "workspace_method": workspace_method,
            "network_definition": "explicit_batch",
            "parser_errors": parser_errors,
            "build_log_notes": [
                "TensorRT may cast ONNX INT64 weights down to INT32 during parsing/building.",
                "TensorRT may disable TF32 when the current GPU does not support TF32.",
            ],
        },
        "network": collect_network_io(network),
        "runtime_paths": runtime_meta,
        "versions": {
            "python": platform.python_version(),
            "tensorrt": version_of("tensorrt"),
            "nvidia-cudnn-cu12": version_of("nvidia-cudnn-cu12"),
            "nvidia-cublas-cu12": version_of("nvidia-cublas-cu12"),
            "nvidia-cuda-nvrtc-cu12": version_of("nvidia-cuda-nvrtc-cu12"),
        },
        "known_risks": [
            "tensorrt_8_6_1_manual_zip_install",
            "mx250_pascal_sm61_requires_legacy_tensorrt",
            "fp32_build_only_first_version",
            "engine_is_gpu_and_tensorrt_version_specific",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(),
    }
    metadata_path.write_text(json.dumps(build_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return build_meta


def save_failure_metadata(args: argparse.Namespace, error: Exception) -> None:
    metadata_path = args.metadata.expanduser().resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed",
        "precision": args.precision,
        "onnx_path": str(args.onnx.expanduser().resolve()),
        "engine_path": str(args.engine.expanduser().resolve()),
        "trt_root": str(args.trt_root.expanduser().resolve()),
        "error_type": type(error).__name__,
        "error": str(error),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(),
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        meta = build_engine(args)
    except Exception as exc:
        save_failure_metadata(args, exc)
        raise

    print(
        "TensorRT engine build complete: "
        f"engine={meta['engine']['engine_path']} "
        f"metadata={Path(args.metadata)} "
        f"precision={meta['precision']} "
        f"trt={meta['tensorrt']['version']}"
    )


if __name__ == "__main__":
    main()
