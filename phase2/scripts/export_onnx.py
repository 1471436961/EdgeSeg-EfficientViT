"""Export EfficientViT-Seg-B0 to ONNX for Phase 2.

This script builds the fixed-shape ONNX baseline described in
``phase2/design_notes/onnx_export_design.md``. It exports the Phase 1
PyTorch model at Cityscapes resolution, runs ONNX checker, optionally runs
ONNXRuntime CPU validation, and writes a reproducibility metadata JSON.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from _compat import install_import_compat_patches
from _common import (
    DEFAULT_RESOLUTION,
    parse_resolution,
    repo_root,
    resolve_script_version,
    save_json,
    sha256_of_file,
    sha256_of_tensor,
    version_of,
)


SCRIPT_NAME = "export_onnx.py"
DEFAULT_OPSET = 17
DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 1e-4


def default_output_path(args: argparse.Namespace) -> Path:
    h, w = args.resolution
    return Path(f"phase2/results/onnx/efficientvit_seg_{args.model}_{args.dataset}_{h}x{w}.onnx")


def default_metadata_path(args: argparse.Namespace) -> Path:
    h, w = args.resolution
    return Path(f"phase2/results/metrics/onnx_export_{args.model}_{args.dataset}_{h}x{w}.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export EfficientViT-Seg-B0 to ONNX.")
    p.add_argument("--weights", required=True, help="Path to Cityscapes B0 weights.")
    p.add_argument("--input", "--input-image", dest="input_image", required=True, help="Fixed input image path.")
    p.add_argument("--resolution", type=parse_resolution, default=DEFAULT_RESOLUTION, help="Input HxW, default 1024x2048.")
    p.add_argument("--model", default="b0", help="EfficientViT-Seg variant, default b0.")
    p.add_argument("--dataset", default="cityscapes", help="Dataset suffix, default cityscapes.")
    p.add_argument("--output", type=Path, default=None, help="ONNX output path.")
    p.add_argument("--metadata", type=Path, default=None, help="Metadata JSON output path.")
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET, help=f"ONNX opset, default {DEFAULT_OPSET}.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    p.add_argument("--skip-ort", action="store_true", help="Skip ONNXRuntime validation.")
    p.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    p.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    args = p.parse_args()
    args.output = args.output or default_output_path(args)
    args.metadata = args.metadata or default_metadata_path(args)
    return args


def build_model(args: argparse.Namespace) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    compat_meta = install_import_compat_patches()
    from efficientvit.seg_model_zoo import create_efficientvit_seg_model  # type: ignore

    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"weights file not found: {weights_path}")

    zoo_key = f"efficientvit-seg-{args.model}-{args.dataset}"
    model = create_efficientvit_seg_model(
        name=zoo_key,
        pretrained=True,
        weight_url=str(weights_path),
    )
    model.eval().to(args.device)
    weights_meta = {
        "weights_status": "loaded",
        "weights_path": str(weights_path),
        "weights_sha256": sha256_of_file(weights_path),
        "zoo_key": zoo_key,
        "weights_load_msg": f"create_efficientvit_seg_model(name={zoo_key!r}, pretrained=True) loaded checkpoint",
    }
    return model, weights_meta, compat_meta


def build_input_tensor(args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, Any]]:
    from PIL import Image

    h, w = args.resolution
    image_path = Path(args.input_image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")

    image = Image.open(image_path).convert("RGB").resize((w, h))
    arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).contiguous().to(args.device)
    meta = {
        "input_status": "image",
        "input_path": str(image_path),
        "input_sha256": sha256_of_file(image_path),
        "input_tensor_sha256": sha256_of_tensor(tensor),
        "input_resolution": [h, w],
        "batch_size": 1,
        "dtype": "float32",
        "preprocess": "PIL RGB resize to fixed resolution, scale to [0,1], no mean/std normalization",
    }
    return tensor, meta


def run_pytorch_reference(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        y = model(x)
    if not isinstance(y, torch.Tensor):
        raise TypeError(f"expected tensor output, got {type(y)!r}")
    return y.detach().cpu().numpy()


def export_onnx(model: torch.nn.Module, x: torch.Tensor, output_path: Path, opset: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            x,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["segout"],
            dynamic_axes=None,
        )


def check_onnx_model(path: Path) -> Dict[str, Any]:
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return {
        "onnx_checker_pass": True,
        "ir_version": model.ir_version,
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "graph_nodes": len(model.graph.node),
    }


def run_ort_validation(path: Path, x: torch.Tensor, reference: np.ndarray, atol: float, rtol: float) -> Dict[str, Any]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ort_inputs = {"input": x.detach().cpu().numpy()}
    outputs = sess.run(["segout"], ort_inputs)
    ort_out = outputs[0]
    diff = ort_out.astype(np.float64) - reference.astype(np.float64)
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(reference.astype(np.float64)), 1e-12)
    rel_diff = abs_diff / denom
    ref_flat = reference.reshape(-1).astype(np.float64)
    ort_flat = ort_out.reshape(-1).astype(np.float64)
    norm = np.linalg.norm(ref_flat) * np.linalg.norm(ort_flat)
    cosine = float(np.dot(ref_flat, ort_flat) / norm) if norm > 0 else math.nan
    allclose = bool(np.allclose(ort_out, reference, atol=atol, rtol=rtol))
    return {
        "onnxruntime_ran": True,
        "onnxruntime_providers": sess.get_providers(),
        "output_shape": list(ort_out.shape),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_rel_diff": float(rel_diff.max()),
        "cosine_similarity": cosine,
        "allclose_pass": allclose,
        "atol": atol,
        "rtol": rtol,
    }


def collect_versions() -> Dict[str, Optional[str]]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnx": version_of("onnx"),
        "onnxruntime": version_of("onnxruntime"),
        "numpy": version_of("numpy"),
    }


def assemble_metadata(
    args: argparse.Namespace,
    weights_meta: Dict[str, Any],
    input_meta: Dict[str, Any],
    compat_meta: Dict[str, Any],
    reference: np.ndarray,
    checker_meta: Dict[str, Any],
    validation_meta: Dict[str, Any],
) -> Dict[str, Any]:
    h, w = args.resolution
    output_path = Path(args.output).resolve()
    return {
        "status": "ok",
        "model_name": f"efficientvit_seg_{args.model}",
        "dataset": args.dataset,
        "input_resolution": [h, w],
        "batch_size": 1,
        "dtype": "float32",
        "device": args.device,
        "weights": weights_meta,
        "input": input_meta,
        "onnx": {
            "onnx_path": str(output_path),
            "onnx_sha256": sha256_of_file(output_path),
            "opset": args.opset,
            "input_names": ["input"],
            "output_names": ["segout"],
            "dynamic_axes": None,
            "expected_output_shape": [1, 19, h // 8, w // 8],
            "pytorch_output_shape": list(reference.shape),
        },
        "versions": collect_versions(),
        "compat": compat_meta,
        "validation": {**checker_meta, **validation_meta},
        "known_risks": [
            "fixed_shape_export",
            "bicubic_resize_may_affect_tensorrt",
            "litemla_shape_branch_frozen",
            "phase2_first_version_uses_legacy_torch_onnx_export",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }


def main() -> None:
    args = parse_args()
    model, weights_meta, compat_meta = build_model(args)
    x, input_meta = build_input_tensor(args)
    reference = run_pytorch_reference(model, x)
    export_onnx(model, x, Path(args.output), args.opset)
    checker_meta = check_onnx_model(Path(args.output))
    if args.skip_ort:
        validation_meta = {
            "onnxruntime_ran": False,
            "allclose_pass": None,
            "atol": args.atol,
            "rtol": args.rtol,
        }
    else:
        validation_meta = run_ort_validation(Path(args.output), x, reference, args.atol, args.rtol)

    payload = assemble_metadata(
        args=args,
        weights_meta=weights_meta,
        input_meta=input_meta,
        compat_meta=compat_meta,
        reference=reference,
        checker_meta=checker_meta,
        validation_meta=validation_meta,
    )
    save_json(Path(args.metadata), payload)
    print(
        "ONNX export complete: "
        f"onnx={Path(args.output)} metadata={Path(args.metadata)} "
        f"checker={checker_meta['onnx_checker_pass']} "
        f"ort={validation_meta.get('onnxruntime_ran')} "
        f"allclose={validation_meta.get('allclose_pass')}"
    )


if __name__ == "__main__":
    main()
