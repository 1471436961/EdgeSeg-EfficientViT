"""Validate the Phase 3 relu_linear_att Plugin against a PyTorch reference.

This is a Step 5 single-layer validation script. It deliberately does not do
EfficientViT graph surgery. The goal is narrower: prove that the Plugin's
enqueue path computes the P1a contract [1,384,64,128] -> [1,128,64,128].
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


SCRIPT_NAME = "validate_relu_linear_attention_plugin.py"
DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3
PLUGIN_NAME = "EdgesegReluLinearAttention_TRT"
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "edgeseg"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
PHASE2_SCRIPTS = ROOT / "phase2" / "scripts"
if str(PHASE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PHASE2_SCRIPTS))

from _common import resolve_script_version, save_json, sha256_of_file, sha256_of_tensor, version_of  # noqa: E402
from _trt_runtime import DEFAULT_TRT_ROOT, prepare_runtime_paths  # noqa: E402
from build_plugin_toy_engine import build_toy_engine  # noqa: E402


_DLL_DIRECTORY_HANDLES: List[object] = []


def default_plugin_dll() -> Path:
    return Path("phase3/plugin/build/edgeseg_relu_linear_attention_plugin.dll")


def default_engine_path() -> Path:
    return Path("phase3/results/engines/relu_linear_attention_toy_fp32.engine")


def default_build_metadata_path() -> Path:
    return Path("phase3/results/metrics/relu_linear_attention_toy_build.json")


def default_validation_metadata_path() -> Path:
    return Path("phase3/results/metrics/relu_linear_attention_plugin_validation.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the relu_linear_att Plugin against PyTorch reference.")
    p.add_argument("--plugin-dll", type=Path, default=default_plugin_dll(), help="Plugin DLL path.")
    p.add_argument("--engine", type=Path, default=default_engine_path(), help="Toy engine path.")
    p.add_argument("--build-metadata", type=Path, default=default_build_metadata_path(), help="Toy build metadata path.")
    p.add_argument("--metadata", type=Path, default=default_validation_metadata_path(), help="Validation metadata JSON path.")
    p.add_argument("--trt-root", type=Path, default=DEFAULT_TRT_ROOT, help="TensorRT zip root directory.")
    p.add_argument("--workspace-mib", type=int, default=256, help="Workspace limit when rebuilding toy engine.")
    p.add_argument("--seed", type=int, default=42, help="Fixed random seed for the validation tensor.")
    p.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    p.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    p.add_argument("--no-rebuild-engine", action="store_true", help="Use the existing toy engine instead of rebuilding it.")
    p.add_argument("--verbose", action="store_true", help="Use verbose TensorRT logger.")
    return p.parse_args()


def add_cuda_toolkit_dll_dir(runtime_meta: Dict[str, Any]) -> None:
    cuda_path = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"))
    cuda_bin = cuda_path / "bin"
    if cuda_bin.is_dir():
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(cuda_bin)))
        os.environ["PATH"] = f"{cuda_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        runtime_meta["phase3_extra_dll_dirs_added"] = [str(cuda_bin)]
        runtime_meta["phase3_extra_dll_dirs_missing"] = []
    else:
        runtime_meta["phase3_extra_dll_dirs_added"] = []
        runtime_meta["phase3_extra_dll_dirs_missing"] = [str(cuda_bin)]


def relu_linear_attention_reference(qkv, dim: int = 16, eps: float = 1.0e-15):
    import torch
    import torch.nn.functional as F

    batch, _, height, width = list(qkv.size())
    qkv = qkv.float().reshape(batch, -1, 3 * dim, height * width)
    q = qkv[:, :, 0:dim]
    k = qkv[:, :, dim : 2 * dim]
    v = qkv[:, :, 2 * dim :]
    q = torch.relu(q)
    k = torch.relu(k)
    v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
    vk = torch.matmul(v, k.transpose(-1, -2))
    out = torch.matmul(vk, q)
    out = out[:, :, :-1] / (out[:, :, -1:] + eps)
    return out.reshape(batch, -1, height, width)


def shape_tuple(shape) -> Tuple[int, ...]:
    return tuple(int(shape[i]) for i in range(len(shape)))


def binding_dtype_to_torch(trt, dtype):
    import torch

    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.int32:
        return torch.int32
    raise TypeError(f"unsupported TensorRT binding dtype: {dtype}")


def allocate_bindings(trt, engine, context, input_tensor):
    import torch

    bindings: List[int] = [0] * int(engine.num_bindings)
    output_tensor = None
    binding_meta: List[Dict[str, Any]] = []
    for index in range(engine.num_bindings):
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning)
            name = engine.get_binding_name(index)
            is_input = bool(engine.binding_is_input(index))
            dtype = engine.get_binding_dtype(index)
            shape = shape_tuple(context.get_binding_shape(index))
        torch_dtype = binding_dtype_to_torch(trt, dtype)
        if is_input:
            if shape != tuple(input_tensor.shape):
                raise ValueError(f"input shape mismatch: engine={shape}, tensor={tuple(input_tensor.shape)}")
            if input_tensor.dtype != torch_dtype:
                raise ValueError(f"input dtype mismatch: engine={torch_dtype}, tensor={input_tensor.dtype}")
            bindings[index] = int(input_tensor.contiguous().data_ptr())
        else:
            output_tensor = torch.empty(shape, dtype=torch_dtype, device=input_tensor.device)
            bindings[index] = int(output_tensor.data_ptr())
        binding_meta.append(
            {
                "index": int(index),
                "name": str(name),
                "is_input": is_input,
                "shape": list(shape),
                "dtype": str(dtype),
            }
        )
    if output_tensor is None:
        raise RuntimeError("engine has no output binding")
    return bindings, output_tensor, binding_meta


def compare_outputs(plugin_output, reference, atol: float, rtol: float) -> Dict[str, Any]:
    plugin_np = plugin_output.detach().cpu().numpy().astype(np.float64)
    ref_np = reference.detach().cpu().numpy().astype(np.float64)
    diff = plugin_np - ref_np
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / np.maximum(np.abs(ref_np), 1e-12)
    plugin_flat = plugin_np.reshape(-1)
    ref_flat = ref_np.reshape(-1)
    norm = np.linalg.norm(plugin_flat) * np.linalg.norm(ref_flat)
    cosine = float(np.dot(plugin_flat, ref_flat) / norm) if norm > 0 else math.nan
    plugin_argmax = np.argmax(plugin_np, axis=1)
    ref_argmax = np.argmax(ref_np, axis=1)
    return {
        "comparison_target": "pytorch_relu_linear_att_reference",
        "output_shape": list(plugin_np.shape),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_rel_diff": float(rel_diff.max()),
        "cosine_similarity": cosine,
        "allclose_pass": bool(np.allclose(plugin_np, ref_np, atol=atol, rtol=rtol)),
        "allclose_pass_atol_1e_2_rtol_1e_2": bool(np.allclose(plugin_np, ref_np, atol=1e-2, rtol=1e-2)),
        "argmax_pixel_agreement": float(np.mean(plugin_argmax == ref_argmax)),
        "argmax_mismatch_pixels": int(np.sum(plugin_argmax != ref_argmax)),
        "argmax_total_pixels": int(ref_argmax.size),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    plugin_dll = args.plugin_dll.expanduser().resolve()
    if not plugin_dll.is_file():
        raise FileNotFoundError(f"Plugin DLL not found: {plugin_dll}")

    if not args.no_rebuild_engine:
        build_args = argparse.Namespace(
            plugin_dll=args.plugin_dll,
            engine=args.engine,
            metadata=args.build_metadata,
            trt_root=args.trt_root,
            workspace_mib=args.workspace_mib,
            verbose=args.verbose,
        )
        build_toy_engine(build_args)

    runtime_meta = prepare_runtime_paths(args.trt_root.expanduser().resolve())
    add_cuda_toolkit_dll_dir(runtime_meta)

    try:
        import tensorrt as trt
    except Exception as exc:
        raise RuntimeError(f"failed to import TensorRT after DLL path setup: {exc}") from exc

    dll_handle = ctypes.CDLL(str(plugin_dll))
    severity = trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO
    logger = trt.Logger(severity)
    trt.init_libnvinfer_plugins(logger, "")

    engine_path = args.engine.expanduser().resolve()
    if not engine_path.is_file():
        raise FileNotFoundError(f"Toy engine not found: {engine_path}")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize toy engine: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("failed to create execution context")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    qkv = torch.randn((1, 384, 64, 128), device="cuda", dtype=torch.float32).contiguous()
    with torch.inference_mode():
        reference = relu_linear_attention_reference(qkv)

    bindings, output_tensor, binding_meta = allocate_bindings(trt, engine, context, qkv)
    stream = torch.cuda.current_stream()
    ok = context.execute_async_v2(bindings=bindings, stream_handle=int(stream.cuda_stream))
    if not ok:
        raise RuntimeError("TensorRT execute_async_v2 returned False")
    stream.synchronize()

    comparison = compare_outputs(output_tensor, reference, args.atol, args.rtol)
    payload = {
        "status": "ok",
        "purpose": "phase3_step5_relu_linear_attention_plugin_single_layer_validation",
        "mx250_constraints": {
            "target_sm": "sm_61",
            "precision": "fp32",
            "tensor_cores": "not_available_on_mx250_pascal",
            "workspace_policy": "small_workspace_only_vk_8x17x16_float32",
        },
        "plugin": {
            "name": PLUGIN_NAME,
            "version": PLUGIN_VERSION,
            "namespace": PLUGIN_NAMESPACE,
            "dll_path": str(plugin_dll),
            "dll_sha256": sha256_of_file(plugin_dll),
        },
        "engine": {
            "engine_path": str(engine_path),
            "engine_sha256": sha256_of_file(engine_path),
            "engine_size_bytes": engine_path.stat().st_size,
        },
        "input": {
            "shape": list(qkv.shape),
            "dtype": str(qkv.dtype),
            "seed": int(args.seed),
            "sha256": sha256_of_tensor(qkv),
        },
        "reference": {
            "name": "pytorch_relu_linear_attention_reference",
            "dim": 16,
            "eps": 1.0e-15,
        },
        "bindings": binding_meta,
        "comparison": comparison,
        "runtime_paths": runtime_meta,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tensorrt": version_of("tensorrt"),
            "numpy": version_of("numpy"),
        },
        "notes": [
            "This validates only the single Plugin layer contract; it does not replace the full EfficientViT graph.",
            "The CUDA implementation uses two kernels: compute vk workspace, then compute normalized output.",
            "Numerical differences are expected to be nonzero because reduction order differs from torch.matmul/cuBLAS.",
        ],
        "_dll_handle_alive": bool(dll_handle),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }
    save_json(args.metadata, payload)
    return payload


def save_failure_metadata(args: argparse.Namespace, error: Exception) -> None:
    payload = {
        "status": "failed",
        "purpose": "phase3_step5_relu_linear_attention_plugin_single_layer_validation",
        "plugin_dll": str(args.plugin_dll.expanduser().resolve()),
        "engine_path": str(args.engine.expanduser().resolve()),
        "error_type": type(error).__name__,
        "error": str(error),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }
    save_json(args.metadata.expanduser().resolve(), payload)


def main() -> None:
    args = parse_args()
    try:
        payload = validate(args)
    except Exception as exc:
        save_failure_metadata(args, exc)
        raise
    cmp = payload["comparison"]
    print(
        "Plugin validation complete: "
        f"metadata={args.metadata} "
        f"max_abs_diff={cmp['max_abs_diff']:.6g} "
        f"mean_abs_diff={cmp['mean_abs_diff']:.6g} "
        f"cosine={cmp['cosine_similarity']:.9f} "
        f"allclose={cmp['allclose_pass']}"
    )


if __name__ == "__main__":
    main()
