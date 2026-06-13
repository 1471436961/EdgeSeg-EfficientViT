"""Benchmark Phase 3 Plugin engine against the Phase 2 TensorRT baseline.

This script is the Phase 3 Step 7 endpoint: it loads the custom Plugin DLL,
deserializes both the baseline TensorRT FP32 engine and the Plugin FP32 engine,
then measures both engines with the same CUDA Event protocol used in Phase 2.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


SCRIPT_NAME = "benchmark_plugin_engine.py"
DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 1e-4
DEFAULT_BASELINE_ENGINE = Path("phase2/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_fp32.engine")
DEFAULT_PLUGIN_ENGINE = Path(
    "phase3/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_relu_linear_att_plugin_fp32.engine"
)
DEFAULT_METADATA = Path("phase3/results/metrics/relu_linear_attention_plugin_engine_benchmark.json")
PLUGIN_NAME = "EdgesegReluLinearAttention_TRT"
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "edgeseg"
_PLUGIN_DLL_HANDLES: List[object] = []


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
PHASE2_SCRIPTS = ROOT / "phase2" / "scripts"
if str(PHASE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PHASE2_SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark_trt_engine as phase2_bench  # noqa: E402
from _common import (  # noqa: E402
    DEFAULT_RESOLUTION,
    parse_resolution,
    resolve_script_version,
    save_json,
    sha256_of_file,
    sha256_of_tensor,
    version_of,
)
from _trt_runtime import DEFAULT_TRT_ROOT, prepare_runtime_paths  # noqa: E402
from build_plugin_toy_engine import (  # noqa: E402
    add_phase3_runtime_dirs,
    creator_names,
    default_plugin_dll,
)
from export_onnx import build_input_tensor, build_model, run_pytorch_reference  # noqa: E402


phase2_bench.torch = torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Phase 3 Plugin engine against Phase 2 TensorRT baseline.")
    parser.add_argument("--baseline-engine", type=Path, default=DEFAULT_BASELINE_ENGINE)
    parser.add_argument("--plugin-engine", type=Path, default=DEFAULT_PLUGIN_ENGINE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--plugin-dll", type=Path, default=default_plugin_dll())
    parser.add_argument("--trt-root", type=Path, default=DEFAULT_TRT_ROOT)
    parser.add_argument("--weights", required=True, help="Path to Cityscapes B0 weights for PyTorch reference.")
    parser.add_argument("--input", "--input-image", dest="input_image", required=True, help="Fixed input image path.")
    parser.add_argument("--resolution", type=parse_resolution, default=DEFAULT_RESOLUTION)
    parser.add_argument("--model", default="b0")
    parser.add_argument("--dataset", default="cityscapes")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measure", type=int, default=100)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument(
        "--benchmark-target",
        choices=["both", "baseline", "plugin"],
        default="both",
        help="Which engine(s) to execute. Step 7 uses both; Step 8 Nsight attribution should use plugin.",
    )
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--nvtx", action="store_true", help="Annotate execute ranges for optional Nsight debugging.")
    return parser.parse_args()


def register_plugin_runtime(args: argparse.Namespace) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
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
    _PLUGIN_DLL_HANDLES.append(dll_handle)

    logger = trt.Logger(trt.Logger.WARNING)
    if hasattr(trt, "init_libnvinfer_plugins"):
        trt.init_libnvinfer_plugins(logger, "")

    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator(PLUGIN_NAME, PLUGIN_VERSION, PLUGIN_NAMESPACE)
    registered_creators = creator_names(registry)
    if creator is None:
        raise RuntimeError(
            f"Plugin creator not found: name={PLUGIN_NAME}, version={PLUGIN_VERSION}, namespace={PLUGIN_NAMESPACE}"
        )

    plugin_meta = {
        "name": PLUGIN_NAME,
        "version": PLUGIN_VERSION,
        "namespace": PLUGIN_NAMESPACE,
        "dll_path": str(plugin_dll),
        "dll_sha256": sha256_of_file(plugin_dll),
        "creator_found": True,
        "registered_creator_count": len(registered_creators),
    }
    return trt, runtime_meta, plugin_meta


def deserialize_engine(trt, engine_path: Path):
    resolved = engine_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"engine file not found: {resolved}")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(resolved.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize TensorRT engine: {resolved}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"failed to create TensorRT execution context: {resolved}")
    return resolved, runtime, engine, context


def engine_meta(engine_path: Path, engine) -> Dict[str, Any]:
    return {
        "engine_path": str(engine_path),
        "engine_sha256": sha256_of_file(engine_path),
        "engine_size_bytes": engine_path.stat().st_size,
        "num_bindings": int(engine.num_bindings),
        "num_layers": int(engine.num_layers),
        "has_implicit_batch_dimension": bool(engine.has_implicit_batch_dimension),
    }


def output_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def compare_arrays(candidate: np.ndarray, reference: np.ndarray, target: str, atol: float, rtol: float) -> Dict[str, Any]:
    diff = candidate.astype(np.float64) - reference.astype(np.float64)
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(reference.astype(np.float64)), 1e-12)
    rel_diff = abs_diff / denom
    cand_flat = candidate.reshape(-1).astype(np.float64)
    ref_flat = reference.reshape(-1).astype(np.float64)
    norm = np.linalg.norm(cand_flat) * np.linalg.norm(ref_flat)
    cosine = float(np.dot(cand_flat, ref_flat) / norm) if norm > 0 else math.nan
    cand_argmax = np.argmax(candidate, axis=1)
    ref_argmax = np.argmax(reference, axis=1)
    return {
        "comparison_target": target,
        "output_shape": list(candidate.shape),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_rel_diff": float(rel_diff.max()),
        "cosine_similarity": cosine,
        "allclose_pass": bool(np.allclose(candidate, reference, atol=atol, rtol=rtol)),
        "allclose_pass_atol_1e_3_rtol_1e_3": bool(np.allclose(candidate, reference, atol=1e-3, rtol=1e-3)),
        "argmax_pixel_agreement": float(np.mean(cand_argmax == ref_argmax)),
        "argmax_mismatch_pixels": int(np.sum(cand_argmax != ref_argmax)),
        "argmax_total_pixels": int(ref_argmax.size),
        "atol": atol,
        "rtol": rtol,
    }


def benchmark_one_engine(trt, label: str, engine, context, x: torch.Tensor, args: argparse.Namespace) -> Dict[str, Any]:
    bindings, binding_meta, output_tensor = phase2_bench.allocate_bindings(trt, engine, context, x)
    times, stream_handle = phase2_bench.measure_latency_ms(context, bindings, args.warmup, args.measure, args.nvtx)
    torch.cuda.synchronize()
    output_np = output_to_numpy(output_tensor)
    return {
        "label": label,
        "bindings": binding_meta["bindings"],
        "timing": {
            "mode": "latency",
            "clock": "cuda_events",
            "scope": "engine_execute_only_no_preprocess_no_h2d_no_d2h",
            "warmup": int(args.warmup),
            "measure": int(args.measure),
            "stream_handle": int(stream_handle),
            "nvtx_enabled": bool(args.nvtx),
            "latency_ms": phase2_bench.summarize_times(times),
            "samples_ms": times,
        },
        "output": {
            "shape": list(output_np.shape),
            "sha256": sha256_of_tensor(output_tensor),
        },
        "_output_np": output_np,
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator != 0 else math.nan


def save_failure_metadata(args: argparse.Namespace, error: Exception) -> None:
    payload = {
        "status": "failed",
        "purpose": "phase3_step7_plugin_engine_benchmark",
        "benchmark_target": args.benchmark_target,
        "baseline_engine": str(args.baseline_engine.expanduser().resolve()),
        "plugin_engine": str(args.plugin_engine.expanduser().resolve()),
        "plugin_dll": str(args.plugin_dll.expanduser().resolve()),
        "error_type": type(error).__name__,
        "error": str(error),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }
    save_json(args.metadata.expanduser().resolve(), payload)


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    trt, runtime_meta, plugin_meta = register_plugin_runtime(args)
    baseline_path = baseline_runtime = baseline_engine = baseline_context = None
    plugin_path = plugin_runtime = plugin_engine = plugin_context = None
    if args.benchmark_target in ("both", "baseline"):
        baseline_path, baseline_runtime, baseline_engine, baseline_context = deserialize_engine(trt, args.baseline_engine)
    if args.benchmark_target in ("both", "plugin"):
        plugin_path, plugin_runtime, plugin_engine, plugin_context = deserialize_engine(trt, args.plugin_engine)

    x, input_meta = build_input_tensor(args)
    if not x.is_cuda:
        raise RuntimeError("Plugin engine benchmark requires CUDA input tensor")
    x = x.contiguous()

    reference = None
    weights_meta = None
    compat_meta = None
    if not args.skip_reference:
        model, weights_meta, compat_meta = build_model(args)
        reference = run_pytorch_reference(model, x)

    torch.cuda.reset_peak_memory_stats()
    baseline_result = None
    plugin_result = None
    baseline_np = None
    plugin_np = None
    execution_order: List[str] = []
    if baseline_engine is not None and baseline_context is not None:
        baseline_result = benchmark_one_engine(
            trt, "phase2_tensorrt_fp32_baseline", baseline_engine, baseline_context, x, args
        )
        baseline_np = baseline_result.pop("_output_np")
        execution_order.append("baseline")
    if plugin_engine is not None and plugin_context is not None:
        plugin_result = benchmark_one_engine(
            trt, "phase3_relu_linear_att_plugin_fp32", plugin_engine, plugin_context, x, args
        )
        plugin_np = plugin_result.pop("_output_np")
        execution_order.append("plugin")

    comparisons: Dict[str, Any] = {}
    if baseline_np is not None and plugin_np is not None:
        comparisons["plugin_vs_baseline_trt"] = compare_arrays(
            plugin_np, baseline_np, "phase2_tensorrt_fp32_baseline", args.atol, args.rtol
        )
    else:
        comparisons["plugin_vs_baseline_trt"] = {"comparison_target": "skipped", "allclose_pass": None}

    if reference is not None and baseline_np is not None:
        comparisons["baseline_trt_vs_pytorch"] = compare_arrays(
            baseline_np, reference, "pytorch_cuda", args.atol, args.rtol
        )
    else:
        comparisons["baseline_trt_vs_pytorch"] = {"comparison_target": "skipped", "allclose_pass": None}

    if reference is not None and plugin_np is not None:
        comparisons["plugin_trt_vs_pytorch"] = compare_arrays(plugin_np, reference, "pytorch_cuda", args.atol, args.rtol)
    else:
        comparisons["plugin_trt_vs_pytorch"] = {"comparison_target": "skipped", "allclose_pass": None}

    speedup: Dict[str, Any]
    if baseline_result is not None and plugin_result is not None:
        baseline_p50 = baseline_result["timing"]["latency_ms"]["p50"]
        plugin_p50 = plugin_result["timing"]["latency_ms"]["p50"]
        baseline_mean = baseline_result["timing"]["latency_ms"]["mean"]
        plugin_mean = plugin_result["timing"]["latency_ms"]["mean"]
        speedup = {
            "baseline": "phase2_tensorrt_fp32_baseline",
            "candidate": "phase3_relu_linear_att_plugin_fp32",
            "p50_speedup_baseline_over_plugin": safe_ratio(baseline_p50, plugin_p50),
            "mean_speedup_baseline_over_plugin": safe_ratio(baseline_mean, plugin_mean),
            "p50_delta_ms_plugin_minus_baseline": float(plugin_p50 - baseline_p50),
            "mean_delta_ms_plugin_minus_baseline": float(plugin_mean - baseline_mean),
            "plugin_is_faster_by_p50": bool(plugin_p50 < baseline_p50),
            "plugin_is_faster_by_mean": bool(plugin_mean < baseline_mean),
        }
    else:
        speedup = {"status": "skipped", "reason": f"benchmark_target={args.benchmark_target}"}

    # Keep runtimes alive until after all output metadata is assembled.
    runtime_keepalive = bool(baseline_runtime or plugin_runtime)

    return {
        "status": "ok",
        "purpose": "phase3_step7_plugin_engine_benchmark",
        "scope": "end_to_end_efficientvit_fp32_baseline_vs_stage2_relu_linear_att_plugin",
        "precision": "fp32",
        "benchmark_target": args.benchmark_target,
        "input": {
            **input_meta,
            "input_tensor_sha256_after_cuda": sha256_of_tensor(x),
        },
        "weights": weights_meta,
        "compat": compat_meta,
        "plugin": plugin_meta,
        "engines": {
            "baseline": engine_meta(baseline_path, baseline_engine) if baseline_engine is not None else None,
            "plugin": engine_meta(plugin_path, plugin_engine) if plugin_engine is not None else None,
        },
        "runtime_paths": runtime_meta,
        "tensorrt": {
            "version": trt.__version__,
            "plugin_creator_required": {
                "name": PLUGIN_NAME,
                "version": PLUGIN_VERSION,
                "namespace": PLUGIN_NAMESPACE,
            },
        },
        "benchmark": {
            "protocol": "same_process_same_input_cuda_events_execute_only",
            "execution_order": execution_order,
            "baseline": baseline_result,
            "plugin": plugin_result,
            "speedup": speedup,
        },
        "comparisons": comparisons,
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
        },
        "known_risks": [
            "plugin_engine_replaces_tensorRT_internal_implementation_for_target_subgraph",
            "latency_excludes_preprocess_h2d_d2h",
            "execution_order_is_baseline_then_plugin_order_bias_possible",
            "engine_is_specific_to_gpu_and_tensorrt_version",
        ],
        "_runtime_keepalive": runtime_keepalive,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }


def write_summary(metadata_path: Path, payload: Dict[str, Any]) -> Path:
    summary_path = metadata_path.with_name(metadata_path.stem + "_summary.md")
    baseline_entry = payload["benchmark"].get("baseline")
    plugin_entry = payload["benchmark"].get("plugin")
    baseline = baseline_entry["timing"]["latency_ms"] if baseline_entry else None
    plugin = plugin_entry["timing"]["latency_ms"] if plugin_entry else None
    speedup = payload["benchmark"]["speedup"]
    plugin_vs_baseline = payload["comparisons"]["plugin_vs_baseline_trt"]
    plugin_vs_pt = payload["comparisons"]["plugin_trt_vs_pytorch"]
    lines = [
        "# ReLU Linear Attention Plugin Engine Benchmark Summary",
        "",
        f"- Benchmark target: `{payload.get('benchmark_target', 'both')}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Baseline TRT p50 | {baseline['p50']:.4f} ms |" if baseline else "| Baseline TRT p50 | skipped |",
        f"| Plugin TRT p50 | {plugin['p50']:.4f} ms |" if plugin else "| Plugin TRT p50 | skipped |",
        (
            f"| p50 delta (plugin - baseline) | {speedup['p50_delta_ms_plugin_minus_baseline']:.4f} ms |"
            if "p50_delta_ms_plugin_minus_baseline" in speedup
            else "| p50 delta (plugin - baseline) | skipped |"
        ),
        (
            f"| p50 speedup (baseline / plugin) | {speedup['p50_speedup_baseline_over_plugin']:.4f}x |"
            if "p50_speedup_baseline_over_plugin" in speedup
            else "| p50 speedup (baseline / plugin) | skipped |"
        ),
        f"| Baseline TRT mean | {baseline['mean']:.4f} ms |" if baseline else "| Baseline TRT mean | skipped |",
        f"| Plugin TRT mean | {plugin['mean']:.4f} ms |" if plugin else "| Plugin TRT mean | skipped |",
        (
            f"| mean delta (plugin - baseline) | {speedup['mean_delta_ms_plugin_minus_baseline']:.4f} ms |"
            if "mean_delta_ms_plugin_minus_baseline" in speedup
            else "| mean delta (plugin - baseline) | skipped |"
        ),
        (
            f"| mean speedup (baseline / plugin) | {speedup['mean_speedup_baseline_over_plugin']:.4f}x |"
            if "mean_speedup_baseline_over_plugin" in speedup
            else "| mean speedup (baseline / plugin) | skipped |"
        ),
        "",
        "## Correctness",
        "",
        "| Comparison | allclose | max abs diff | mean abs diff | argmax agreement |",
        "|---|---:|---:|---:|---:|",
        (
            "| Plugin TRT vs Baseline TRT | "
            f"{plugin_vs_baseline['allclose_pass']} | "
            f"{plugin_vs_baseline.get('max_abs_diff', float('nan')):.6g} | "
            f"{plugin_vs_baseline.get('mean_abs_diff', float('nan')):.6g} | "
            f"{plugin_vs_baseline.get('argmax_pixel_agreement', float('nan')):.6f} |"
        ),
        (
            "| Plugin TRT vs PyTorch | "
            f"{plugin_vs_pt.get('allclose_pass')} | "
            f"{plugin_vs_pt.get('max_abs_diff', float('nan')):.6g} | "
            f"{plugin_vs_pt.get('mean_abs_diff', float('nan')):.6g} | "
            f"{plugin_vs_pt.get('argmax_pixel_agreement', float('nan')):.6f} |"
        ),
        "",
        "## Interpretation",
        "",
        "- `>1.0x` means the Plugin engine is faster than the Phase 2 TensorRT FP32 baseline.",
        "- `<1.0x` means the first Plugin kernel is slower end-to-end and should be treated as an integration/correctness milestone, not a performance win.",
        "- This benchmark excludes preprocessing, H2D/D2H, and output postprocess, matching the Phase 2 execute-only latency protocol.",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main() -> None:
    args = parse_args()
    try:
        payload = build_payload(args)
    except Exception as exc:
        save_failure_metadata(args, exc)
        raise

    metadata_path = args.metadata.expanduser().resolve()
    save_json(metadata_path, payload)
    summary_path = write_summary(metadata_path, payload)
    plugin_entry = payload["benchmark"].get("plugin")
    baseline_entry = payload["benchmark"].get("baseline")
    speedup = payload["benchmark"].get("speedup", {})
    baseline_p50_text = (
        f"{baseline_entry['timing']['latency_ms']['p50']:.3f}ms" if baseline_entry else "skipped"
    )
    plugin_p50_text = f"{plugin_entry['timing']['latency_ms']['p50']:.3f}ms" if plugin_entry else "skipped"
    speedup_text = (
        f"{speedup['p50_speedup_baseline_over_plugin']:.3f}x"
        if "p50_speedup_baseline_over_plugin" in speedup
        else "skipped"
    )
    print(
        "Plugin engine benchmark complete: "
        f"metadata={metadata_path} "
        f"summary={summary_path} "
        f"target={payload.get('benchmark_target')} "
        f"baseline_p50={baseline_p50_text} "
        f"plugin_p50={plugin_p50_text} "
        f"speedup={speedup_text} "
        f"plugin_vs_baseline_allclose={payload['comparisons']['plugin_vs_baseline_trt']['allclose_pass']}"
    )


if __name__ == "__main__":
    main()
