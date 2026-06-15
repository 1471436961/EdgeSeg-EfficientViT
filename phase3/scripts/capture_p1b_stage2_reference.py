"""Capture PyTorch block-level references for the P1b Plugin boundary.

This script does not run TensorRT and does not validate the current P1b
skeleton Plugin. Its job is narrower: capture the two real stage2/context
LiteMLA blocks and save the tensors needed to validate a future
aggregation + cat + relu_linear_att CUDA implementation.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


SCRIPT_NAME = "capture_p1b_stage2_reference.py"
TARGET_MODULES = (
    "backbone.stages.2.op_list.1.context_module.main",
    "backbone.stages.2.op_list.2.context_module.main",
)


def phase2_scripts_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "phase2" / "scripts"


PHASE2_SCRIPTS = phase2_scripts_dir()
if str(PHASE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PHASE2_SCRIPTS))

from _common import (  # noqa: E402
    DEFAULT_RESOLUTION,
    parse_resolution,
    repo_root,
    resolve_script_version,
    save_json,
    sha256_of_file,
    sha256_of_tensor,
    version_of,
)
from _compat import install_import_compat_patches  # noqa: E402


def default_weights_path() -> Path:
    return Path("phase1/weights/efficientvit_seg_b0_cityscapes.pt")


def default_input_path() -> Path:
    return Path("phase1/data/city_asset_cityscapes_like.png")


def default_metadata_path() -> Path:
    return Path("phase3/results/metrics/p1b_stage2_reference_capture.json")


def default_tensor_bundle_path() -> Path:
    return Path("phase3/results/tensors/p1b_stage2_reference_capture.npz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture P1b stage2/context PyTorch reference tensors.")
    p.add_argument("--weights", type=Path, default=default_weights_path(), help="EfficientViT-Seg-B0 weights path.")
    p.add_argument("--input-image", type=Path, default=default_input_path(), help="Input image path.")
    p.add_argument("--use-dummy-input", action="store_true", help="Use deterministic random input instead of image.")
    p.add_argument("--dummy-seed", type=int, default=42, help="Dummy input seed.")
    p.add_argument("--resolution", type=parse_resolution, default=DEFAULT_RESOLUTION, help="Input HxW.")
    p.add_argument("--model", default="b0", help="EfficientViT-Seg variant.")
    p.add_argument("--dataset", default="cityscapes", help="Dataset suffix.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    p.add_argument("--metadata", type=Path, default=default_metadata_path(), help="Output metadata JSON.")
    p.add_argument("--tensor-bundle", type=Path, default=default_tensor_bundle_path(), help="Output NPZ tensor bundle.")
    p.add_argument("--no-save-tensors", action="store_true", help="Only write JSON metadata, not the NPZ tensor bundle.")
    return p.parse_args()


def build_model(args: argparse.Namespace) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    compat_meta = install_import_compat_patches()
    from efficientvit.seg_model_zoo import create_efficientvit_seg_model  # type: ignore

    weights_path = args.weights.expanduser().resolve()
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
    h, w = args.resolution
    if args.use_dummy_input:
        gen = torch.Generator(device="cpu").manual_seed(args.dummy_seed)
        tensor = torch.randn((1, 3, h, w), generator=gen, dtype=torch.float32).contiguous().to(args.device)
        meta = {
            "input_status": "dummy",
            "input_path": None,
            "input_sha256": None,
            "input_tensor_sha256": sha256_of_tensor(tensor),
            "input_dummy_seed": int(args.dummy_seed),
            "input_resolution": [h, w],
            "batch_size": 1,
            "dtype": "float32",
            "preprocess": "torch.randn with fixed CPU seed",
        }
        return tensor, meta

    from PIL import Image

    image_path = args.input_image.expanduser().resolve()
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
        "input_dummy_seed": None,
        "input_resolution": [h, w],
        "batch_size": 1,
        "dtype": "float32",
        "preprocess": "PIL RGB resize to fixed resolution, scale to [0,1], no mean/std normalization",
    }
    return tensor, meta


def find_target_modules(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    modules = dict(model.named_modules())
    missing = [name for name in TARGET_MODULES if name not in modules]
    if missing:
        raise KeyError(f"target LiteMLA modules not found: {missing}")
    return {name: modules[name] for name in TARGET_MODULES}


def tensor_stats(t: torch.Tensor) -> Dict[str, Any]:
    detached = t.detach()
    as_float = detached.float()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "contiguous": bool(detached.is_contiguous()),
        "sha256": sha256_of_tensor(detached),
        "min": float(as_float.min().item()),
        "max": float(as_float.max().item()),
        "mean": float(as_float.mean().item()),
        "std": float(as_float.std(unbiased=False).item()),
    }


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    diff = (a.detach().float() - b.detach().float()).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "allclose_atol_1e_5_rtol_1e_5": bool(torch.allclose(a, b, atol=1e-5, rtol=1e-5)),
    }


def capture_context_inputs(
    model: torch.nn.Module,
    target_modules: Dict[str, torch.nn.Module],
    model_input: torch.Tensor,
) -> Dict[str, Dict[str, torch.Tensor]]:
    captures: Dict[str, Dict[str, torch.Tensor]] = {}
    handles = []

    def make_hook(name: str):
        def _hook(module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            captures[name] = {
                "context_input": inputs[0].detach().clone().contiguous(),
                "module_output": output.detach().clone().contiguous(),
            }

        return _hook

    for name, module in target_modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.inference_mode():
            _ = model(model_input)
    finally:
        for handle in handles:
            handle.remove()

    missing = [name for name in TARGET_MODULES if name not in captures]
    if missing:
        raise RuntimeError(f"target module hooks did not fire: {missing}")
    return captures


def compute_block_reference(
    module: torch.nn.Module,
    captured: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    context_input = captured["context_input"]
    captured_module_output = captured["module_output"]

    with torch.inference_mode():
        qkv = module.qkv(context_input).contiguous()
        aggregated_qkv = module.aggreg[0](qkv).contiguous()
        cat_qkv = torch.cat([qkv, aggregated_qkv], dim=1).contiguous()
        attention_out = module.relu_linear_att(cat_qkv).to(cat_qkv.dtype).contiguous()
        projected_attention = module.proj(attention_out).contiguous()

    depthwise = module.aggreg[0][0]
    pointwise = module.aggreg[0][1]
    if depthwise.bias is not None or pointwise.bias is not None:
        raise ValueError("P1b first version assumes aggregation convs have no bias")

    tensors = {
        "context_input": context_input,
        "qkv": qkv,
        "aggregated_qkv": aggregated_qkv,
        "cat_qkv": cat_qkv,
        "attention_out": attention_out,
        "projected_attention": projected_attention,
        "captured_module_output": captured_module_output,
        "depthwise_weight": depthwise.weight.detach().clone().contiguous(),
        "pointwise_weight": pointwise.weight.detach().clone().contiguous(),
    }

    meta = {
        "module_type": type(module).__name__,
        "dim": int(module.dim),
        "eps": float(module.eps),
        "aggregation_count": int(len(module.aggreg)),
        "weights": {
            "depthwise": {
                **tensor_stats(tensors["depthwise_weight"]),
                "groups": int(depthwise.groups),
                "kernel_size": list(depthwise.kernel_size),
                "padding": list(depthwise.padding),
                "bias": None,
            },
            "pointwise": {
                **tensor_stats(tensors["pointwise_weight"]),
                "groups": int(pointwise.groups),
                "kernel_size": list(pointwise.kernel_size),
                "padding": list(pointwise.padding),
                "bias": None,
            },
        },
        "tensors": {
            key: tensor_stats(value)
            for key, value in tensors.items()
            if key not in {"depthwise_weight", "pointwise_weight"}
        },
        "proj_check_against_module_output": compare_tensors(projected_attention, captured_module_output),
    }
    return tensors, meta


def assert_contract(block_meta: Dict[str, Any]) -> None:
    expected_shapes = {
        "context_input": [1, 64, 64, 128],
        "qkv": [1, 192, 64, 128],
        "aggregated_qkv": [1, 192, 64, 128],
        "cat_qkv": [1, 384, 64, 128],
        "attention_out": [1, 128, 64, 128],
    }
    for name, shape in expected_shapes.items():
        actual = block_meta["tensors"][name]["shape"]
        if actual != shape:
            raise ValueError(f"{name} shape mismatch: expected {shape}, got {actual}")
    if block_meta["weights"]["depthwise"]["shape"] != [192, 1, 5, 5]:
        raise ValueError("depthwise weight shape mismatch")
    if block_meta["weights"]["depthwise"]["groups"] != 192:
        raise ValueError("depthwise groups mismatch")
    if block_meta["weights"]["pointwise"]["shape"] != [192, 16, 1, 1]:
        raise ValueError("pointwise weight shape mismatch")
    if block_meta["weights"]["pointwise"]["groups"] != 12:
        raise ValueError("pointwise groups mismatch")
    if not block_meta["proj_check_against_module_output"]["allclose_atol_1e_5_rtol_1e_5"]:
        raise ValueError("projected attention does not reproduce captured LiteMLA module output")


def save_tensor_bundle(path: Path, block_tensors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    arrays: Dict[str, np.ndarray] = {}
    for block_name, tensors in block_tensors.items():
        short = block_name.replace("backbone.stages.2.", "stage2.").replace(".", "_")
        for tensor_name in ["qkv", "attention_out", "depthwise_weight", "pointwise_weight"]:
            arrays[f"{short}__{tensor_name}"] = tensors[tensor_name].detach().cpu().contiguous().numpy()

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        "tensor_bundle_saved": True,
        "tensor_bundle_path": str(path.resolve()),
        "tensor_bundle_size_bytes": path.resolve().stat().st_size,
        "tensor_bundle_arrays": sorted(arrays.keys()),
        "tensor_bundle_note": "Generated artifact for local validation; not intended for git tracking.",
    }


def collect_versions() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": version_of("numpy"),
        "pillow": version_of("Pillow"),
    }


def capture(args: argparse.Namespace) -> Dict[str, Any]:
    model, weights_meta, compat_meta = build_model(args)
    x, input_meta = build_input_tensor(args)
    target_modules = find_target_modules(model)
    captured_inputs = capture_context_inputs(model, target_modules, x)

    block_payloads: List[Dict[str, Any]] = []
    block_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
    for module_id, name in enumerate(TARGET_MODULES):
        tensors, meta = compute_block_reference(target_modules[name], captured_inputs[name])
        assert_contract(meta)
        block_tensors[name] = tensors
        block_payloads.append(
            {
                "module_id": module_id,
                "name": name,
                "semantic": "stage2/context LiteMLA P1b boundary",
                **meta,
            }
        )

    bundle_meta: Dict[str, Any]
    if args.no_save_tensors:
        bundle_meta = {
            "tensor_bundle_saved": False,
            "tensor_bundle_path": None,
            "tensor_bundle_note": "Skipped by --no-save-tensors.",
        }
    else:
        bundle_meta = save_tensor_bundle(args.tensor_bundle.expanduser(), block_tensors)

    return {
        "status": "ok",
        "purpose": "p1b_stage2_block_reference_capture",
        "model_name": f"efficientvit_seg_{args.model}",
        "dataset": args.dataset,
        "input_resolution": list(args.resolution),
        "device": args.device,
        "target_modules": list(TARGET_MODULES),
        "weights": weights_meta,
        "input": input_meta,
        "compat": compat_meta,
        "blocks": block_payloads,
        "tensor_bundle": bundle_meta,
        "versions": collect_versions(),
        "known_risks": [
            "p1b_skeleton_zero_fill_not_validated_here",
            "fp32_only_reference",
            "fixed_cityscapes_resolution_only",
            "tensor_bundle_is_local_large_artifact",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }


def main() -> None:
    args = parse_args()
    payload = capture(args)
    save_json(args.metadata.expanduser(), payload)
    print(
        "P1b stage2 reference captured: "
        f"metadata={args.metadata} blocks={len(payload['blocks'])} "
        f"tensor_bundle={payload['tensor_bundle'].get('tensor_bundle_path')}"
    )


if __name__ == "__main__":
    main()
