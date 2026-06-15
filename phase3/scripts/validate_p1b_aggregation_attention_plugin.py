"""Validate the P1b aggregation + relu_linear_att Plugin against PyTorch reference.

This is a block-level validation script. It uses the tensors captured by
capture_p1b_stage2_reference.py and builds one toy TensorRT engine per target
stage2/context block, with that block's real aggregation weights embedded as
ONNX initializers.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


SCRIPT_NAME = "validate_p1b_aggregation_attention_plugin.py"
PLUGIN_NAME = "EdgesegAggregationReluLinearAttention_TRT"
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "edgeseg"
DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3


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


def default_reference_metadata() -> Path:
    return Path("phase3/results/metrics/p1b_stage2_reference_capture.json")


def default_tensor_bundle() -> Path:
    return Path("phase3/results/tensors/p1b_stage2_reference_capture.npz")


def default_metadata_path() -> Path:
    return Path("phase3/results/metrics/p1b_aggregation_attention_plugin_validation.json")


def default_onnx_dir() -> Path:
    return Path("phase3/results/onnx")


def default_engine_dir() -> Path:
    return Path("phase3/results/engines")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate P1b Plugin against captured PyTorch block references.")
    parser.add_argument("--plugin-dll", type=Path, default=default_plugin_dll(), help="Plugin DLL path.")
    parser.add_argument("--reference-metadata", type=Path, default=default_reference_metadata())
    parser.add_argument("--tensor-bundle", type=Path, default=default_tensor_bundle())
    parser.add_argument("--onnx-dir", type=Path, default=default_onnx_dir())
    parser.add_argument("--engine-dir", type=Path, default=default_engine_dir())
    parser.add_argument("--metadata", type=Path, default=default_metadata_path())
    parser.add_argument("--trt-root", type=Path, default=DEFAULT_TRT_ROOT)
    parser.add_argument("--workspace-mib", type=int, default=256)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def short_block_key(module_name: str) -> str:
    return module_name.replace("backbone.stages.2.", "stage2.").replace(".", "_")


def load_reference(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    reference_path = args.reference_metadata.expanduser().resolve()
    bundle_path = args.tensor_bundle.expanduser().resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"reference metadata not found: {reference_path}")
    if not bundle_path.is_file():
        raise FileNotFoundError(f"tensor bundle not found: {bundle_path}")

    metadata = json.loads(reference_path.read_text(encoding="utf-8"))
    bundle = np.load(bundle_path)
    arrays = {name: bundle[name] for name in bundle.files}
    return metadata, arrays


def make_block_onnx(path: Path, block_key: str, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    qkv_name = f"{block_key}__qkv"
    depthwise_name = f"{block_key}__depthwise_weight"
    pointwise_name = f"{block_key}__pointwise_weight"
    missing = [name for name in [qkv_name, depthwise_name, pointwise_name] if name not in arrays]
    if missing:
        raise KeyError(f"missing arrays for {block_key}: {missing}")

    qkv = helper.make_tensor_value_info("qkv", TensorProto.FLOAT, [1, 192, 64, 128])
    output = helper.make_tensor_value_info("attention_out", TensorProto.FLOAT, [1, 128, 64, 128])
    depthwise = numpy_helper.from_array(arrays[depthwise_name].astype(np.float32), name="aggregation_depthwise_weight")
    pointwise = numpy_helper.from_array(arrays[pointwise_name].astype(np.float32), name="aggregation_pointwise_weight")

    plugin_node = helper.make_node(
        PLUGIN_NAME,
        inputs=["qkv", "aggregation_depthwise_weight", "aggregation_pointwise_weight"],
        outputs=["attention_out"],
        name=f"{block_key}__p1b_plugin",
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
        f"{block_key}__p1b_validation_graph",
        [qkv],
        [output],
        initializer=[depthwise, pointwise],
    )
    model = helper.make_model(
        graph,
        producer_name="edgeseg_phase3_p1b_validation",
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
        "onnx_path": str(path),
        "onnx_sha256": sha256_of_file(path),
        "onnx_size_bytes": path.stat().st_size,
        "checker": checker,
    }


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
            contiguous = input_tensor.contiguous()
            bindings[index] = int(contiguous.data_ptr())
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
        "output_shape": list(plugin_np.shape),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_rel_diff": float(rel_diff.max()),
        "cosine_similarity": cosine,
        "allclose_pass": bool(np.allclose(plugin_np, ref_np, atol=atol, rtol=rtol)),
        "allclose_pass_atol_1e_2_rtol_1e_2": bool(np.allclose(plugin_np, ref_np, atol=1e-2, rtol=1e-2)),
        "argmax_channel_agreement": float(np.mean(plugin_argmax == ref_argmax)),
        "argmax_mismatch_pixels": int(np.sum(plugin_argmax != ref_argmax)),
        "argmax_total_pixels": int(ref_argmax.size),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def build_engine_from_onnx(trt, logger, onnx_path: Path, engine_path: Path, workspace_mib: int):
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    parse_ok = parser.parse(onnx_path.read_bytes())
    parser_errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not parse_ok:
        raise RuntimeError("TensorRT ONNX parser failed:\n" + "\n".join(parser_errors))

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_mib) * 1024 * 1024
    workspace_method = set_workspace_limit(config, trt, workspace_bytes)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT builder returned None serialized P1b validation engine")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    return {
        "engine_path": str(engine_path),
        "engine_sha256": sha256_of_file(engine_path),
        "engine_size_bytes": engine_path.stat().st_size,
        "parser_errors": parser_errors,
        "workspace_mib": int(workspace_mib),
        "workspace_bytes": workspace_bytes,
        "workspace_method": workspace_method,
        "network": {
            "inputs": [
                {
                    "name": network.get_input(i).name,
                    "shape": dims_to_list(network.get_input(i).shape),
                    "dtype": str(network.get_input(i).dtype),
                }
                for i in range(network.num_inputs)
            ],
            "outputs": [
                {
                    "name": network.get_output(i).name,
                    "shape": dims_to_list(network.get_output(i).shape),
                    "dtype": str(network.get_output(i).dtype),
                }
                for i in range(network.num_outputs)
            ],
            "num_layers": int(network.num_layers),
        },
    }


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    reference_meta, arrays = load_reference(args)
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

    block_results = []
    for block in reference_meta["blocks"]:
        module_name = block["name"]
        block_key = short_block_key(module_name)
        qkv = torch.from_numpy(arrays[f"{block_key}__qkv"].astype(np.float32)).contiguous().cuda()
        reference = torch.from_numpy(arrays[f"{block_key}__attention_out"].astype(np.float32)).contiguous().cuda()

        onnx_path = args.onnx_dir.expanduser().resolve() / f"p1b_validation_{block_key}.onnx"
        engine_path = args.engine_dir.expanduser().resolve() / f"p1b_validation_{block_key}_fp32.engine"
        onnx_meta = make_block_onnx(onnx_path, block_key, arrays)
        engine_meta = build_engine_from_onnx(trt, logger, onnx_path, engine_path, args.workspace_mib)

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"failed to deserialize validation engine: {engine_path}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("failed to create TensorRT execution context")

        bindings, output_tensor, binding_meta = allocate_bindings(trt, engine, context, qkv)
        ok = context.execute_async_v2(bindings=bindings, stream_handle=torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError(f"TensorRT execute_async_v2 returned false for {module_name}")
        torch.cuda.synchronize()
        comparison = compare_outputs(output_tensor, reference, args.atol, args.rtol)

        block_results.append(
            {
                "module_id": block["module_id"],
                "name": module_name,
                "block_key": block_key,
                "reference_tensor_sha256": block["tensors"]["attention_out"]["sha256"],
                "qkv_tensor_sha256": block["tensors"]["qkv"]["sha256"],
                "depthwise_weight_sha256": block["weights"]["depthwise"]["sha256"],
                "pointwise_weight_sha256": block["weights"]["pointwise"]["sha256"],
                "onnx": onnx_meta,
                "engine": engine_meta,
                "bindings": binding_meta,
                "comparison": comparison,
            }
        )

    overall_pass = all(result["comparison"]["allclose_pass"] for result in block_results)
    return {
        "status": "ok" if overall_pass else "failed",
        "purpose": "p1b_aggregation_attention_plugin_block_validation",
        "overall_pass": overall_pass,
        "plugin": {
            "name": PLUGIN_NAME,
            "version": PLUGIN_VERSION,
            "namespace": PLUGIN_NAMESPACE,
            "dll_path": str(plugin_dll),
            "dll_sha256": sha256_of_file(plugin_dll),
            "creator_found": True,
            "registered_creator_count": len(registered_creators),
        },
        "reference": {
            "metadata_path": str(args.reference_metadata.expanduser().resolve()),
            "tensor_bundle_path": str(args.tensor_bundle.expanduser().resolve()),
            "target_modules": reference_meta.get("target_modules"),
            "input": reference_meta.get("input"),
            "weights": reference_meta.get("weights"),
        },
        "blocks": block_results,
        "runtime_paths": runtime_meta,
        "versions": {
            "python": platform.python_version(),
            "tensorrt": trt.__version__,
            "torch": torch.__version__,
            "numpy": version_of("numpy"),
            "onnx": version_of("onnx"),
        },
        "notes": [
            "This validates block-level P1b math against captured PyTorch attention_out.",
            "Each block gets its own toy ONNX/engine because aggregation weights differ.",
            "This is not an end-to-end EfficientViT latency benchmark.",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
        "_dll_handle_alive": bool(dll_handle),
    }


def main() -> None:
    args = parse_args()
    payload = validate(args)
    payload.pop("_dll_handle_alive", None)
    save_json(args.metadata.expanduser(), payload)
    print(
        "P1b validation complete: "
        f"metadata={args.metadata} overall_pass={payload['overall_pass']} "
        f"blocks={len(payload['blocks'])}"
    )


if __name__ == "__main__":
    main()
