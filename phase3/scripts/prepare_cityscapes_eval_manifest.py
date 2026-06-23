"""Prepare a reproducible Cityscapes evaluation manifest for Phase 3.

The script does not download Cityscapes because the dataset is license-gated.
Instead it validates a local Cityscapes tree and writes a JSON manifest used by
``evaluate_cityscapes_miou.py``. The expected layout is:

    CITYSCAPES_ROOT/
      leftImg8bit/val/<city>/*_leftImg8bit.png
      gtFine/val/<city>/*_gtFine_labelIds.png

Large image/label files should remain outside git. ``phase3/data`` is ignored
except for its ``.gitkeep`` placeholder.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_NAME = "prepare_cityscapes_eval_manifest.py"
DEFAULT_OUTPUT = Path("phase3/results/metrics/cityscapes_val_manifest.json")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
PHASE2_SCRIPTS = ROOT / "phase2" / "scripts"
if str(PHASE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PHASE2_SCRIPTS))

from _common import resolve_script_version, save_json, sha256_of_file  # noqa: E402


def default_cityscapes_root() -> Path:
    env_value = os.environ.get("CITYSCAPES_ROOT")
    if env_value:
        return Path(env_value)
    return Path("phase3/data/cityscapes")


def stable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Cityscapes data and write a Phase 3 manifest.")
    parser.add_argument("--cityscapes-root", type=Path, default=default_cityscapes_root())
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--label-kind", default="labelIds", choices=["labelIds", "labelTrainIds"])
    parser.add_argument("--limit", type=int, default=None, help="Optional fixed prefix subset size.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write a status=missing JSON instead of failing when data is absent.",
    )
    return parser.parse_args()


def label_path_for(image_path: Path, root: Path, split: str, label_kind: str) -> Path:
    rel = image_path.relative_to(root / "leftImg8bit" / split)
    label_name = image_path.name.replace("_leftImg8bit.png", f"_gtFine_{label_kind}.png")
    return root / "gtFine" / split / rel.parent / label_name


def collect_samples(root: Path, split: str, label_kind: str, limit: int | None) -> List[Dict[str, Any]]:
    image_root = root / "leftImg8bit" / split
    if not image_root.is_dir():
        raise FileNotFoundError(f"Cityscapes image directory not found: {image_root}")

    image_paths = sorted(image_root.glob("*/*_leftImg8bit.png"))
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError(f"No Cityscapes images found under: {image_root}")

    samples: List[Dict[str, Any]] = []
    missing_labels: List[str] = []
    for idx, image_path in enumerate(image_paths):
        label_path = label_path_for(image_path, root, split, label_kind)
        if not label_path.is_file():
            missing_labels.append(str(label_path))
            continue
        city = image_path.parent.name
        sample_id = image_path.name.replace("_leftImg8bit.png", "")
        samples.append(
            {
                "index": idx,
                "city": city,
                "sample_id": sample_id,
                "image_path": stable_path(image_path),
                "label_path": stable_path(label_path),
                "image_sha256": sha256_of_file(image_path),
                "label_sha256": sha256_of_file(label_path),
            }
        )

    if missing_labels:
        preview = "\n".join(missing_labels[:10])
        raise FileNotFoundError(f"Missing {len(missing_labels)} label files. First entries:\n{preview}")
    return samples


def missing_payload(args: argparse.Namespace, error: Exception) -> Dict[str, Any]:
    root = args.cityscapes_root.expanduser().resolve()
    return {
        "status": "missing",
        "purpose": "phase3_cityscapes_eval_manifest",
        "cityscapes_root": stable_path(root),
        "expected_layout": {
            "images": str(root / "leftImg8bit" / args.split / "<city>" / "*_leftImg8bit.png"),
            "labels": str(root / "gtFine" / args.split / "<city>" / f"*_gtFine_{args.label_kind}.png"),
        },
        "reason": str(error),
        "next_action": [
            "Download Cityscapes leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip with a Cityscapes account.",
            "Extract both archives into the same CITYSCAPES_ROOT directory.",
            "Run this script again without --allow-missing.",
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
    }


def main() -> None:
    args = parse_args()
    root = args.cityscapes_root.expanduser().resolve()
    try:
        samples = collect_samples(root, args.split, args.label_kind, args.limit)
        payload = {
            "status": "ok",
            "purpose": "phase3_cityscapes_eval_manifest",
            "cityscapes_root": stable_path(root),
            "split": args.split,
            "label_kind": args.label_kind,
            "sample_count": len(samples),
            "limit": args.limit,
            "samples": samples,
            "versions": {
                "python": platform.python_version(),
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "script_version": resolve_script_version(SCRIPT_NAME, Path(__file__)),
        }
    except Exception as exc:
        if not args.allow_missing:
            raise
        payload = missing_payload(args, exc)

    output = args.output.expanduser().resolve()
    save_json(output, payload)
    print(
        "Cityscapes manifest check complete: "
        f"status={payload['status']} output={output} samples={payload.get('sample_count', 0)}"
    )


if __name__ == "__main__":
    main()
