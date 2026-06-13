"""Microbenchmark the Phase 3 relu_linear_att Plugin against PyTorch.

This Step 5.5 script measures only the single-layer P1a contract:
[1,384,64,128] -> [1,128,64,128]. It does not do EfficientViT graph surgery.
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


SCRIPT_NAME = "benchmark_relu_linear_attention_plugin.py"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
PHASE2_SCRIPTS = ROOT / "phase2" / "scripts"
if str(PHASE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PHASE2_SCRIPTS))

from _common import resolve_script_version, save_json, sha256_of_file, sha256_of_tensor, version_of  # noqa: E402
from _trt_runtime import DEFAULT_TRT_ROOT, prepare_runtime_paths  # noqa: E402
from build_plugin_toy_engine import build_toy_engine  # noqa: E402
from validate_relu_linear_attention_plugin import (  # noqa: E402
    add_cuda_toolkit_dll_dir,
    allocate_bindings,
    compare_outputs,
    default_build_metadata_path,
    default_engine_path,
    default_plugin_dll,
    relu_linear_attention_reference,
)


def default_metadata_path() -> Path:
    return Path("phase3/results/metrics/relu_linear_attention_plugin_microbenchmark.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Microbenchmark the relu_linear_att Plugin single layer.")
    p.add_argument("--plugin-dll", type=Path, default=default_plugin_dll(), help="Plugin DLL path.")
    p.add_argument("--engine", type=Path, default=default_engine_path(), help="Toy engine path.")
    p.add_argument("--build-metadata", type=Path, default=default_build_metadata_path(), help="Toy build metadata path.")
    p.add_argument("--metadata", type=Path, default=default_metadata_path(), help="Output metadata JSON path.")
    p.add_argument("--trt-root", type=Path, default=DEFAULT_TRT_ROOT, help="TensorRT zip root directory.")
    p.add_argument("--workspace-mib", type=int, default=256, help="Workspace limit when rebuilding toy engine.")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--measure", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--no-rebuild-engine", action="store_true", help="Use existing toy engine.")
    p.add_argument("--nvtx", action="store_true", help="Emit NVTX ranges for Nsight Systems.")
    p.add_argument("--verbose", action="store_true", help="Use verbose TensorRT logger.")
    return p.parse_args()


def nvtx_push(torch, name: str, enabled: bool) -> None:
    if enabled:
        torch.cuda.nvtx.range_push(name)


def nvtx_pop(torch, enabled: bool) -> None:
    if enabled:
        torch.cuda.nvtx.range_pop()


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


def execute_plugin_once(context, bindings: List[int], stream_handle: int) -> None:
    ok = context.execute_async_v2(bindings=bindings, stream_handle=stream_handle)
    if not ok:
        raise RuntimeError("TensorRT execute_async_v2 returned False")


def measure_plugin(torch, context, bindings: List[int], warmup: int, measure: int, nvtx: bool) -> List[float]:
    stream = torch.cuda.current_stream()
    stream_handle = int(stream.cuda_stream)

    torch.cuda.synchronize()
    nvtx_push(torch, "plugin_single/warmup", nvtx)
    try:
        for _ in range(warmup):
            nvtx_push(torch, "plugin_single/execute", nvtx)
            try:
                execute_plugin_once(context, bindings, stream_handle)
            finally:
                nvtx_pop(torch, nvtx)
    finally:
        nvtx_pop(torch, nvtx)
    torch.cuda.synchronize()

    times: List[float] = []
    nvtx_push(torch, "plugin_single/measure", nvtx)
    try:
        for _ in range(measure):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            nvtx_push(torch, "plugin_single/execute", nvtx)
            try:
                execute_plugin_once(context, bindings, stream_handle)
            finally:
                nvtx_pop(torch, nvtx)
            end.record(stream)
            end.synchronize()
            times.append(float(start.elapsed_time(end)))
    finally:
        nvtx_pop(torch, nvtx)
    torch.cuda.synchronize()
    return times


def measure_pytorch(torch, qkv, warmup: int, measure: int, nvtx: bool) -> List[float]:
    stream = torch.cuda.current_stream()

    torch.cuda.synchronize()
    nvtx_push(torch, "pytorch_relu_linear_att/warmup", nvtx)
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                nvtx_push(torch, "pytorch_relu_linear_att/execute", nvtx)
                try:
                    relu_linear_attention_reference(qkv)
                finally:
                    nvtx_pop(torch, nvtx)
    finally:
        nvtx_pop(torch, nvtx)
    torch.cuda.synchronize()

    times: List[float] = []
    nvtx_push(torch, "pytorch_relu_linear_att/measure", nvtx)
    try:
        with torch.inference_mode():
            for _ in range(measure):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                nvtx_push(torch, "pytorch_relu_linear_att/execute", nvtx)
                try:
                    relu_linear_attention_reference(qkv)
                finally:
                    nvtx_pop(torch, nvtx)
                end.record(stream)
                end.synchronize()
                times.append(float(start.elapsed_time(end)))
    finally:
        nvtx_pop(torch, nvtx)
    torch.cuda.synchronize()
    return times


def build_or_load_engine(args: argparse.Namespace) -> None:
    if args.no_rebuild_engine:
        return
    build_args = argparse.Namespace(
        plugin_dll=args.plugin_dll,
        engine=args.engine,
        metadata=args.build_metadata,
        trt_root=args.trt_root,
        workspace_mib=args.workspace_mib,
        verbose=args.verbose,
    )
    build_toy_engine(build_args)


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    build_or_load_engine(args)

    runtime_meta = prepare_runtime_paths(args.trt_root.expanduser().resolve())
    add_cuda_toolkit_dll_dir(runtime_meta)
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
    execute_plugin_once(context, bindings, int(stream.cuda_stream))
    stream.synchronize()
    comparison = compare_outputs(output_tensor, reference, args.atol, args.rtol)

    torch.cuda.reset_peak_memory_stats()
    plugin_times = measure_plugin(torch, context, bindings, args.warmup, args.measure, args.nvtx)
    plugin_memory = {
        "torch_max_memory_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        "torch_max_memory_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 * 1024)),
    }

    torch.cuda.reset_peak_memory_stats()
    pytorch_times = measure_pytorch(torch, qkv, args.warmup, args.measure, args.nvtx)
    pytorch_memory = {
        "torch_max_memory_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        "torch_max_memory_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 * 1024)),
    }

    plugin_summary = summarize_times(plugin_times)
    pytorch_summary = summarize_times(pytorch_times)
    speedup = pytorch_summary["p50"] / plugin_summary["p50"] if plugin_summary["p50"] > 0 else None

    payload = {
        "status": "ok",
        "purpose": "phase3_step5_5_relu_linear_attention_plugin_microbenchmark",
        "scope": "single_layer_toy_engine_no_efficientvit_graph_surgery",
        "mx250_constraints": {
            "target_sm": "sm_61",
            "precision": "fp32",
            "tensor_cores": "not_available_on_mx250_pascal",
            "workspace_policy": "small_workspace_only_vk_8x17x16_float32",
        },
        "plugin": {
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
        "bindings": binding_meta,
        "comparison": comparison,
        "timing": {
            "clock": "cuda_events",
            "warmup": int(args.warmup),
            "measure": int(args.measure),
            "nvtx_enabled": bool(args.nvtx),
            "plugin_single_layer_ms": plugin_summary,
            "pytorch_reference_ms": pytorch_summary,
            "plugin_vs_pytorch_p50_speedup": speedup,
            "plugin_samples_ms": plugin_times,
            "pytorch_samples_ms": pytorch_times,
        },
        "memory": {
            "plugin": plugin_memory,
            "pytorch_reference": pytorch_memory,
            "note": "PyTorch allocator stats do not include all TensorRT internal allocations.",
        },
        "runtime_paths": runtime_meta,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tensorrt": version_of("tensorrt"),
            "numpy": version_of("numpy"),
        },
        "notes": [
            "This is a single-layer microbenchmark, not an end-to-end EfficientViT benchmark.",
            "Plugin timing measures TensorRT toy engine execute_async_v2 for one Plugin layer.",
            "PyTorch reference timing includes PyTorch ops and temporary tensor allocations for relu_linear_att.",
            "Use --nvtx under nsys to inspect launch gaps and kernel grouping.",
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
        "purpose": "phase3_step5_5_relu_linear_attention_plugin_microbenchmark",
        "error_type": type(error).__name__,
        "error": str(error),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }
    save_json(args.metadata.expanduser().resolve(), payload)


def main() -> None:
    args = parse_args()
    try:
        payload = run_benchmark(args)
    except Exception as exc:
        save_failure_metadata(args, exc)
        raise

    plugin = payload["timing"]["plugin_single_layer_ms"]
    pytorch = payload["timing"]["pytorch_reference_ms"]
    speedup = payload["timing"]["plugin_vs_pytorch_p50_speedup"]
    print(
        "Plugin microbenchmark complete: "
        f"metadata={args.metadata} "
        f"plugin_p50={plugin['p50']:.4f}ms "
        f"pytorch_p50={pytorch['p50']:.4f}ms "
        f"speedup={speedup:.3f}x "
        f"allclose={payload['comparison']['allclose_pass']}"
    )


if __name__ == "__main__":
    main()
