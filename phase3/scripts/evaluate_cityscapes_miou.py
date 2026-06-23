"""Evaluate Cityscapes mIoU for TensorRT baseline and Phase 3 Plugin engines.

This is an accuracy gate, not a latency benchmark. It reuses the TensorRT
binding-buffer approach from Phase 2/3, upsamples the H/8 ``segout`` logits to
the Cityscapes label resolution with bicubic interpolation, then computes mIoU
over the 19 official Cityscapes trainId classes.
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
from PIL import Image


SCRIPT_NAME = "evaluate_cityscapes_miou.py"
DEFAULT_BASELINE_ENGINE = Path("phase2/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_fp32.engine")
DEFAULT_PLUGIN_ENGINE = Path(
    "phase3/results/engines/efficientvit_seg_b0_cityscapes_1024x2048_relu_linear_att_plugin_stage2_stage3_fp32.engine"
)
DEFAULT_MANIFEST = Path("phase3/results/metrics/cityscapes_val_manifest.json")
DEFAULT_OUTPUT = Path("phase3/results/metrics/cityscapes_miou_p1a_stage2_stage3.json")
DEFAULT_PLUGIN_NAME = "EdgesegReluLinearAttention_TRT"
DEFAULT_PLUGIN_VERSION = "1"
DEFAULT_PLUGIN_NAMESPACE = "edgeseg"
CITYSCAPES_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]
LABEL_ID_TO_TRAIN_ID = np.array(
    [
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        0,
        1,
        -1,
        -1,
        2,
        3,
        4,
        -1,
        -1,
        -1,
        5,
        -1,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        -1,
        -1,
        16,
        17,
        18,
    ],
    dtype=np.int16,
)
_PLUGIN_DLL_HANDLES: List[object] = []
torch = None
F = None


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
    version_of,
)
from _trt_runtime import DEFAULT_TRT_ROOT, import_tensorrt_after_path_setup, load_serialized_engine  # noqa: E402
from phase3.scripts.build_plugin_toy_engine import (  # noqa: E402
    add_phase3_runtime_dirs,
    creator_names,
    default_plugin_dll,
)


def import_torch_after_tensorrt() -> None:
    """Import torch after TensorRT to avoid Windows DLL load-order conflicts."""
    global torch, F
    if torch is not None:
        return
    import torch as torch_module
    import torch.nn.functional as functional_module

    torch = torch_module
    F = functional_module
    phase2_bench.torch = torch_module



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Cityscapes mIoU for TensorRT engines.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-engine", type=Path, default=DEFAULT_BASELINE_ENGINE)
    parser.add_argument("--plugin-engine", type=Path, default=DEFAULT_PLUGIN_ENGINE)
    parser.add_argument("--plugin-dll", type=Path, default=default_plugin_dll())
    parser.add_argument("--plugin-name", default=DEFAULT_PLUGIN_NAME)
    parser.add_argument("--plugin-version", default=DEFAULT_PLUGIN_VERSION)
    parser.add_argument("--plugin-namespace", default=DEFAULT_PLUGIN_NAMESPACE)
    parser.add_argument("--trt-root", type=Path, default=DEFAULT_TRT_ROOT)
    parser.add_argument("--target", choices=["baseline", "plugin", "both"], default="both")
    parser.add_argument("--resolution", type=parse_resolution, default=DEFAULT_RESOLUTION)
    parser.add_argument(
        "--preprocess",
        choices=["official", "deployment"],
        default="official",
        help="official = ImageNet mean/std normalization; deployment = prior Phase 2/3 [0,1] only.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_manifest(path: Path, max_samples: int | None) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise RuntimeError(f"manifest is not usable: status={payload.get('status')} path={resolved}")
    samples = payload.get("samples", [])
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        samples = samples[:max_samples]
    if not samples:
        raise RuntimeError(f"manifest has no samples: {resolved}")
    payload = dict(payload)
    payload["manifest_path"] = str(resolved)
    payload["samples"] = samples
    payload["sample_count"] = len(samples)
    return payload


def resolve_data_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def register_plugin_runtime(args: argparse.Namespace) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    trt, runtime_meta = import_tensorrt_after_path_setup(args.trt_root)
    add_phase3_runtime_dirs(runtime_meta)

    plugin_dll = args.plugin_dll.expanduser().resolve()
    if not plugin_dll.is_file():
        raise FileNotFoundError(f"Plugin DLL not found: {plugin_dll}")
    dll_handle = ctypes.CDLL(str(plugin_dll))
    _PLUGIN_DLL_HANDLES.append(dll_handle)

    logger = trt.Logger(trt.Logger.WARNING)
    if hasattr(trt, "init_libnvinfer_plugins"):
        trt.init_libnvinfer_plugins(logger, "")
    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator(args.plugin_name, args.plugin_version, args.plugin_namespace)
    if creator is None:
        raise RuntimeError(
            "Plugin creator not found: "
            f"name={args.plugin_name}, version={args.plugin_version}, namespace={args.plugin_namespace}"
        )
    return trt, runtime_meta, {
        "name": args.plugin_name,
        "version": args.plugin_version,
        "namespace": args.plugin_namespace,
        "dll_path": str(plugin_dll),
        "dll_sha256": sha256_of_file(plugin_dll),
        "creator_found": True,
        "registered_creator_count": len(creator_names(registry)),
    }


def load_engine(trt, engine_path: Path) -> Dict[str, Any]:
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
        raise RuntimeError(f"failed to create TensorRT context: {resolved}")
    return {
        "path": resolved,
        "runtime": runtime,
        "engine": engine,
        "context": context,
    }


def resize_rgb(image: Image.Image, resolution: Tuple[int, int]) -> np.ndarray:
    h, w = resolution
    arr = np.asarray(image.convert("RGB"))
    try:
        import cv2  # type: ignore

        return cv2.resize(arr, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    except Exception:
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        return np.asarray(Image.fromarray(arr).resize((w, h), resampling))


def preprocess_image(image_path: Path, resolution: Tuple[int, int], preprocess: str) -> torch.Tensor:
    image = Image.open(image_path)
    arr = resize_rgb(image, resolution).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    if preprocess == "official":
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    return tensor.cuda(non_blocking=False).contiguous()


def load_label(label_path: Path, label_kind: str) -> np.ndarray:
    raw = np.asarray(Image.open(label_path), dtype=np.int64)
    if label_kind == "labelTrainIds":
        label = raw.astype(np.int16)
        label[label == 255] = -1
        label[(label < 0) | (label >= len(CITYSCAPES_CLASSES))] = -1
        return label
    max_label_id = int(raw.max()) if raw.size else 0
    if max_label_id >= len(LABEL_ID_TO_TRAIN_ID):
        raise ValueError(f"labelIds contain unsupported id > 33: {label_path}")
    return LABEL_ID_TO_TRAIN_ID[raw]


def run_engine(trt, runner: Dict[str, Any], input_tensor: torch.Tensor) -> torch.Tensor:
    bindings, _binding_meta, output_tensor = phase2_bench.allocate_bindings(
        trt, runner["engine"], runner["context"], input_tensor
    )
    stream = torch.cuda.current_stream()
    ok = runner["context"].execute_async_v2(bindings=bindings, stream_handle=int(stream.cuda_stream))
    if not ok:
        raise RuntimeError(f"TensorRT execute_async_v2 failed for {runner['path']}")
    stream.synchronize()
    return output_tensor


def logits_to_prediction(logits: torch.Tensor, label_shape: Tuple[int, int]) -> np.ndarray:
    with torch.inference_mode():
        upsampled = F.interpolate(logits, size=label_shape, mode="bicubic", align_corners=False)
        pred = torch.argmax(upsampled, dim=1)[0].detach().cpu().numpy().astype(np.int16)
    return pred


def fast_hist(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    mask = (target >= 0) & (target < num_classes)
    encoded = num_classes * target[mask].astype(np.int64) + pred[mask].astype(np.int64)
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def iou_from_hist(hist: np.ndarray) -> Tuple[np.ndarray, float]:
    diag = np.diag(hist).astype(np.float64)
    union = hist.sum(axis=1).astype(np.float64) + hist.sum(axis=0).astype(np.float64) - diag
    iou = np.full_like(diag, np.nan, dtype=np.float64)
    valid = union > 0
    iou[valid] = diag[valid] / union[valid]
    return iou, float(np.nanmean(iou))


def result_from_hist(label: str, hist: np.ndarray, engine_path: Path) -> Dict[str, Any]:
    per_class, miou = iou_from_hist(hist)
    return {
        "label": label,
        "engine_path": str(engine_path),
        "engine_sha256": sha256_of_file(engine_path),
        "miou": miou,
        "miou_percent": miou * 100.0,
        "per_class_iou": {
            name: (None if math.isnan(float(value)) else float(value))
            for name, value in zip(CITYSCAPES_CLASSES, per_class)
        },
        "confusion_matrix": hist.tolist(),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = load_manifest(args.manifest, args.max_samples)
    plugin_meta = None
    if args.target in ("plugin", "both"):
        trt, runtime_meta, plugin_meta = register_plugin_runtime(args)
    else:
        trt, runtime_meta, _runtime, _engine = load_serialized_engine(args.baseline_engine, args.trt_root)
    import_torch_after_tensorrt()

    runners: Dict[str, Dict[str, Any]] = {}
    if args.target in ("baseline", "both"):
        runners["baseline"] = load_engine(trt, args.baseline_engine)
    if args.target in ("plugin", "both"):
        runners["plugin"] = load_engine(trt, args.plugin_engine)

    histograms = {name: np.zeros((len(CITYSCAPES_CLASSES), len(CITYSCAPES_CLASSES)), dtype=np.int64) for name in runners}
    agreement_total = 0
    agreement_match = 0
    samples_meta: List[Dict[str, Any]] = []
    label_kind = manifest.get("label_kind", "labelIds")

    torch.cuda.synchronize()
    with torch.inference_mode():
        for sample in manifest["samples"]:
            image_path = resolve_data_path(sample["image_path"])
            label_path = resolve_data_path(sample["label_path"])
            target = load_label(label_path, label_kind)
            input_tensor = preprocess_image(image_path, args.resolution, args.preprocess)

            preds: Dict[str, np.ndarray] = {}
            for name, runner in runners.items():
                logits = run_engine(trt, runner, input_tensor)
                pred = logits_to_prediction(logits, target.shape)
                preds[name] = pred
                histograms[name] += fast_hist(pred, target, len(CITYSCAPES_CLASSES))

            if "baseline" in preds and "plugin" in preds:
                valid = target >= 0
                agreement_total += int(valid.sum())
                agreement_match += int(((preds["baseline"] == preds["plugin"]) & valid).sum())

            samples_meta.append(
                {
                    "index": sample["index"],
                    "city": sample.get("city"),
                    "sample_id": sample.get("sample_id"),
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "label_shape": list(target.shape),
                }
            )

    results = {
        name: result_from_hist(name, hist, runners[name]["path"])
        for name, hist in histograms.items()
    }
    if "baseline" in results and "plugin" in results:
        comparison = {
            "baseline": "phase2_tensorrt_fp32_baseline",
            "plugin": "phase3_plugin_fp32",
            "miou_delta_plugin_minus_baseline": results["plugin"]["miou"] - results["baseline"]["miou"],
            "miou_percent_delta_plugin_minus_baseline": (
                results["plugin"]["miou_percent"] - results["baseline"]["miou_percent"]
            ),
            "argmax_pixel_agreement_on_valid_labels": (
                float(agreement_match / agreement_total) if agreement_total else None
            ),
            "argmax_mismatch_pixels_on_valid_labels": int(agreement_total - agreement_match),
            "argmax_total_valid_pixels": int(agreement_total),
        }
    else:
        comparison = {"status": "skipped", "reason": f"target={args.target}"}

    return {
        "status": "ok",
        "purpose": "phase3_cityscapes_miou_accuracy_gate",
        "target": args.target,
        "manifest": {
            "manifest_path": manifest["manifest_path"],
            "cityscapes_root": manifest.get("cityscapes_root"),
            "split": manifest.get("split"),
            "label_kind": label_kind,
            "sample_count": manifest["sample_count"],
        },
        "preprocess": {
            "mode": args.preprocess,
            "resolution": list(args.resolution),
            "image_resize": "cv2.INTER_CUBIC if cv2 is available, otherwise PIL bicubic",
            "normalization": (
                "ImageNet mean/std, matching upstream eval_efficientvit_seg_model.py"
                if args.preprocess == "official"
                else "scale to [0,1], no mean/std, matching prior Phase 2/3 deployment benchmark"
            ),
            "logit_resize_for_eval": "bicubic align_corners=False to label resolution, matching EfficientViT resize() default",
        },
        "plugin": plugin_meta,
        "runtime_paths": runtime_meta,
        "results": results,
        "comparison": comparison,
        "samples": samples_meta,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tensorrt": version_of("tensorrt"),
            "numpy": version_of("numpy"),
        },
        "known_risks": [
            "cityscapes_data_is_license_gated_and_not_committed",
            "official_miou_requires_official_preprocess_mode",
            "deployment_preprocess_mode_is_for_regression_only_not_official_cityscapes_score",
            "engine_output_is_h_over_8_and_logits_are_resized_for_eval",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }


def write_summary(output: Path, payload: Dict[str, Any]) -> Path:
    summary = output.with_name(output.stem + "_summary.md")
    lines = [
        "# Phase 3 Cityscapes mIoU Summary",
        "",
        f"- Status: `{payload['status']}`",
        f"- Target: `{payload['target']}`",
        f"- Samples: `{payload['manifest']['sample_count']}`",
        f"- Preprocess: `{payload['preprocess']['mode']}`",
        "",
        "| Engine | mIoU |",
        "|---|---:|",
    ]
    for name, result in payload["results"].items():
        lines.append(f"| {name} | {result['miou_percent']:.3f}% |")
    comp = payload["comparison"]
    if "miou_percent_delta_plugin_minus_baseline" in comp:
        lines.extend(
            [
                "",
                "## Baseline vs Plugin",
                "",
                f"- mIoU delta (plugin - baseline): `{comp['miou_percent_delta_plugin_minus_baseline']:.6f}` percentage points",
                f"- Argmax agreement on valid labels: `{comp['argmax_pixel_agreement_on_valid_labels']:.8f}`",
                f"- Argmax mismatch pixels: `{comp['argmax_mismatch_pixels_on_valid_labels']} / {comp['argmax_total_valid_pixels']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is an accuracy gate, not an execute-only latency benchmark.",
            "- `official` preprocess uses ImageNet mean/std to match upstream Cityscapes eval.",
            "- `deployment` preprocess matches the earlier Phase 2/3 benchmark input convention and is only a regression check.",
        ]
    )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    payload = evaluate(args)
    output = args.output.expanduser().resolve()
    save_json(output, payload)
    summary = write_summary(output, payload)
    result_text = ", ".join(
        f"{name}_miou={result['miou_percent']:.3f}%"
        for name, result in payload["results"].items()
    )
    print(
        "Cityscapes mIoU evaluation complete: "
        f"output={output} summary={summary} {result_text}"
    )


if __name__ == "__main__":
    main()
