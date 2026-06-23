"""TensorRT runtime path and engine-loading helpers for Phase 2 scripts."""

from __future__ import annotations

import os
import site
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_TRT_ROOT = Path(r"E:\NVIDIA\TensorRT-8.6.1.6")
_DLL_DIRECTORY_HANDLES: List[object] = []


def candidate_runtime_dirs(trt_root: Path) -> List[Path]:
    dirs = [trt_root / "lib", trt_root / "bin"]
    for env_name in ("CUDA_PATH", "CUDA_HOME"):
        env_value = os.environ.get(env_name)
        if env_value:
            dirs.append(Path(env_value) / "bin")
    for site_dir in site.getsitepackages():
        base = Path(site_dir) / "nvidia"
        dirs.extend(
            [
                base / "cuda_runtime" / "bin",
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


def import_tensorrt_after_path_setup(trt_root: Path):
    runtime_meta = prepare_runtime_paths(trt_root.expanduser().resolve())
    try:
        import tensorrt as trt
    except Exception as exc:
        raise RuntimeError(f"failed to import TensorRT after DLL path setup: {exc}") from exc
    return trt, runtime_meta


def load_serialized_engine(
    engine_path: Path,
    trt_root: Path,
    logger_severity: Optional[int] = None,
) -> Tuple[Any, Dict[str, Any], Any, Any]:
    trt, runtime_meta = import_tensorrt_after_path_setup(trt_root)

    engine_path = engine_path.expanduser().resolve()
    if not engine_path.is_file():
        raise FileNotFoundError(f"engine file not found: {engine_path}")

    severity = trt.Logger.WARNING if logger_severity is None else logger_severity
    logger = trt.Logger(severity)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
    return trt, runtime_meta, runtime, engine
