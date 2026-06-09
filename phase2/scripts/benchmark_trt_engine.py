"""Benchmark a fixed-shape TensorRT engine and compare it with PyTorch.

The script measures engine execution latency only: input/output GPU buffers are
allocated before timing, and preprocessing is outside the measured region.
"""

from __future__ import annotations

import argparse
import math
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from _common import (
    DEFAULT_RESOLUTION,
    parse_resolution,
    resolve_script_version,
    save_json,
    sha256_of_file,
    sha256_of_tensor,
    version_of,
)
from _trt_runtime import DEFAULT_TRT_ROOT, load_serialized_engine


SCRIPT_NAME = "benchmark_trt_engine.py"
DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 1e-4
torch = None


def default_engine_path(precision: str) -> Path:
    return Path(f"phase2/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_{precision}.engine")


def default_metadata_path(precision: str) -> Path:
    return Path(f"phase2/results/metrics/trt_benchmark_b0_cityscapes_1024x2048_{precision}.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark a TensorRT FP32/FP16 engine.")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32", help="Engine precision label.")
    p.add_argument("--engine", type=Path, default=None, help="Input TensorRT engine path.")
    p.add_argument("--metadata", type=Path, default=None, help="Benchmark metadata JSON path.")
    p.add_argument("--trt-root", type=Path, default=DEFAULT_TRT_ROOT, help="TensorRT zip root directory.")
    p.add_argument("--weights", required=True, help="Path to Cityscapes B0 weights for PyTorch reference.")
    p.add_argument("--input", "--input-image", dest="input_image", required=True, help="Fixed input image path.")
    p.add_argument("--resolution", type=parse_resolution, default=DEFAULT_RESOLUTION, help="Input HxW, default 1024x2048.")
    p.add_argument("--model", default="b0", help="EfficientViT-Seg variant, default b0.")
    p.add_argument("--dataset", default="cityscapes", help="Dataset suffix, default cityscapes.")
    p.add_argument("--device", default="cuda", choices=["cuda"])
    p.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    p.add_argument("--measure", type=int, default=100, help="Measured iterations.")
    p.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    p.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    p.add_argument("--skip-reference", action="store_true", help="Skip PyTorch reference and output comparison.")
    p.add_argument("--nvtx", action="store_true", help="Annotate warmup/measure/execute ranges for Nsight Systems.")
    args = p.parse_args()
    args.engine = args.engine or default_engine_path(args.precision)
    args.metadata = args.metadata or default_metadata_path(args.precision)
    return args


def load_engine(engine_path: Path, trt_root: Path):
    trt, runtime_meta, runtime, engine = load_serialized_engine(engine_path, trt_root)
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("failed to create TensorRT execution context")
    return trt, runtime_meta, runtime, engine, context


def binding_shape_to_tuple(shape) -> Tuple[int, ...]:
    return tuple(int(shape[i]) for i in range(len(shape)))


def binding_dtype_to_torch(trt, dtype):
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int8:
        return torch.int8
    if dtype == trt.bool:
        return torch.bool
    raise TypeError(f"unsupported TensorRT binding dtype: {dtype}")


def allocate_bindings(trt, engine, context, input_tensor: torch.Tensor) -> Tuple[List[int], Dict[str, Any], torch.Tensor]:
    bindings: List[int] = [0] * int(engine.num_bindings)
    binding_meta: List[Dict[str, Any]] = []
    output_tensor = None

    for i in range(engine.num_bindings):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            name = engine.get_binding_name(i)
            is_input = bool(engine.binding_is_input(i))
            dtype = engine.get_binding_dtype(i)
            shape = binding_shape_to_tuple(context.get_binding_shape(i))
        torch_dtype = binding_dtype_to_torch(trt, dtype)

        if is_input:
            if shape != tuple(input_tensor.shape):
                raise ValueError(f"input binding shape mismatch: engine={shape}, tensor={tuple(input_tensor.shape)}")
            if input_tensor.dtype != torch_dtype:
                raise ValueError(f"input binding dtype mismatch: engine={torch_dtype}, tensor={input_tensor.dtype}")
            if not input_tensor.is_contiguous():
                input_tensor = input_tensor.contiguous()
            bindings[i] = int(input_tensor.data_ptr())
        else:
            output_tensor = torch.empty(shape, dtype=torch_dtype, device=input_tensor.device)
            bindings[i] = int(output_tensor.data_ptr())

        binding_meta.append(
            {
                "index": i,
                "name": name,
                "is_input": is_input,
                "shape": list(shape),
                "dtype": str(dtype),
            }
        )

    if output_tensor is None:
        raise RuntimeError("engine has no output binding")
    return bindings, {"bindings": binding_meta}, output_tensor


def nvtx_push(name: str, enabled: bool) -> None:
    if enabled:
        torch.cuda.nvtx.range_push(name)


def nvtx_pop(enabled: bool) -> None:
    if enabled:
        torch.cuda.nvtx.range_pop()


def execute_once(context, bindings: List[int], stream_handle: int, nvtx_name: str = "", nvtx_enabled: bool = False) -> None:
    if nvtx_name:
        nvtx_push(nvtx_name, nvtx_enabled)
    ok = context.execute_async_v2(bindings=bindings, stream_handle=stream_handle)
    if nvtx_name:
        nvtx_pop(nvtx_enabled)
    if not ok:
        raise RuntimeError("TensorRT execute_async_v2 returned False")


def measure_latency_ms(
    context, bindings: List[int], warmup: int, measure: int, nvtx_enabled: bool
) -> Tuple[List[float], float]:
    stream = torch.cuda.current_stream()
    stream_handle = int(stream.cuda_stream)

    torch.cuda.synchronize()
    with torch.inference_mode():
        nvtx_push("trt/warmup", nvtx_enabled)
        try:
            for _ in range(warmup):
                execute_once(context, bindings, stream_handle, "trt/execute", nvtx_enabled)
        finally:
            nvtx_pop(nvtx_enabled)
    torch.cuda.synchronize()

    times: List[float] = []
    with torch.inference_mode():
        nvtx_push("trt/measure", nvtx_enabled)
        try:
            for _ in range(measure):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                execute_once(context, bindings, stream_handle, "trt/execute", nvtx_enabled)
                end.record(stream)
                end.synchronize()
                times.append(float(start.elapsed_time(end)))
        finally:
            nvtx_pop(nvtx_enabled)

    torch.cuda.synchronize()
    return times, stream_handle


def summarize_times(times: List[float]) -> Dict[str, float]:
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def compare_outputs(trt_output: torch.Tensor, reference: np.ndarray, atol: float, rtol: float) -> Dict[str, Any]:
    trt_np = trt_output.detach().cpu().numpy()
    diff = trt_np.astype(np.float64) - reference.astype(np.float64)
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(reference.astype(np.float64)), 1e-12)
    rel_diff = abs_diff / denom
    ref_flat = reference.reshape(-1).astype(np.float64)
    trt_flat = trt_np.reshape(-1).astype(np.float64)
    norm = np.linalg.norm(ref_flat) * np.linalg.norm(trt_flat)
    cosine = float(np.dot(ref_flat, trt_flat) / norm) if norm > 0 else math.nan
    trt_argmax = np.argmax(trt_np, axis=1)
    ref_argmax = np.argmax(reference, axis=1)
    return {
        "comparison_target": "pytorch_cuda",
        "output_shape": list(trt_np.shape),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_rel_diff": float(rel_diff.max()),
        "cosine_similarity": cosine,
        "allclose_pass": bool(np.allclose(trt_np, reference, atol=atol, rtol=rtol)),
        "allclose_pass_atol_1e_3_rtol_1e_3": bool(np.allclose(trt_np, reference, atol=1e-3, rtol=1e-3)),
        "argmax_pixel_agreement": float(np.mean(trt_argmax == ref_argmax)),
        "argmax_mismatch_pixels": int(np.sum(trt_argmax != ref_argmax)),
        "argmax_total_pixels": int(ref_argmax.size),
        "atol": atol,
        "rtol": rtol,
    }


def main() -> None:
    global torch
    args = parse_args()
    trt, runtime_meta, _runtime, engine, context = load_engine(args.engine, args.trt_root)

    import torch as torch_module
    from export_onnx import build_input_tensor, build_model, run_pytorch_reference

    torch = torch_module
    x, input_meta = build_input_tensor(args)
    if not x.is_cuda:
        raise RuntimeError("TensorRT benchmark requires CUDA input tensor")
    x = x.contiguous()

    reference = None
    weights_meta = None
    compat_meta = None
    if not args.skip_reference:
        model, weights_meta, compat_meta = build_model(args)
        reference = run_pytorch_reference(model, x)

    bindings, binding_meta, output_tensor = allocate_bindings(trt, engine, context, x)
    times, stream_handle = measure_latency_ms(context, bindings, args.warmup, args.measure, args.nvtx)
    torch.cuda.synchronize()

    comparison_meta = {"comparison_target": "skipped", "allclose_pass": None, "atol": args.atol, "rtol": args.rtol}
    if reference is not None:
        comparison_meta = compare_outputs(output_tensor, reference, args.atol, args.rtol)

    engine_path = args.engine.expanduser().resolve()
    known_risks = [
        "tensorRT_output_must_be_checked_due_to_int64_to_int32_cast_during_build",
        "latency_excludes_preprocess_h2d_d2h",
        "engine_is_specific_to_gpu_and_tensorrt_version",
    ]
    if args.precision == "fp16":
        known_risks.append("fp16_on_mx250_without_tensor_cores_may_be_slower_than_fp32")

    payload = {
        "status": "ok",
        "precision": args.precision,
        "engine": {
            "engine_path": str(engine_path),
            "engine_sha256": sha256_of_file(engine_path),
            "engine_size_bytes": engine_path.stat().st_size,
        },
        "input": {
            **input_meta,
            "input_tensor_sha256_after_cuda": sha256_of_tensor(x),
        },
        "weights": weights_meta,
        "compat": compat_meta,
        "runtime_paths": runtime_meta,
        "tensorrt": {
            "version": trt.__version__,
            "num_bindings": int(engine.num_bindings),
            "num_layers": int(engine.num_layers),
            "has_implicit_batch_dimension": bool(engine.has_implicit_batch_dimension),
        },
        "bindings": binding_meta["bindings"],
        "timing": {
            "mode": "latency",
            "clock": "cuda_events",
            "scope": "engine_execute_only_no_preprocess_no_h2d_no_d2h",
            "warmup": args.warmup,
            "measure": args.measure,
            "stream_handle": stream_handle,
            "nvtx_enabled": bool(args.nvtx),
            "latency_ms": summarize_times(times),
            "samples_ms": times,
        },
        "comparison": comparison_meta,
        "memory": {
            "scope": "pytorch_cuda_allocator_only_not_tensorrt_internal",
            "torch_max_memory_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
            "torch_max_memory_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 * 1024)),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tensorrt": version_of("tensorrt"),
            "numpy": version_of("numpy"),
            "nvidia-cudnn-cu12": version_of("nvidia-cudnn-cu12"),
            "nvidia-cublas-cu12": version_of("nvidia-cublas-cu12"),
            "nvidia-cuda-nvrtc-cu12": version_of("nvidia-cuda-nvrtc-cu12"),
        },
        "known_risks": known_risks,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }
    save_json(args.metadata, payload)
    print(
        "TensorRT benchmark complete: "
        f"metadata={Path(args.metadata)} "
        f"p50={payload['timing']['latency_ms']['p50']:.3f}ms "
        f"p95={payload['timing']['latency_ms']['p95']:.3f}ms "
        f"allclose={payload['comparison'].get('allclose_pass')}"
    )


if __name__ == "__main__":
    main()
