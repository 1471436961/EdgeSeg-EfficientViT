"""Shared helpers for Phase 2 Python scripts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_RESOLUTION = (1024, 2048)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_resolution(value: str) -> Tuple[int, int]:
    text = value.lower().replace("x", " ").replace(",", " ")
    parts = [p for p in text.split() if p]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must look like 1024x2048")
    try:
        h, w = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution values must be integers") from exc
    if h <= 0 or w <= 0:
        raise argparse.ArgumentTypeError("resolution values must be positive")
    return h, w


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_tensor(t) -> str:
    arr = t.detach().to("cpu").contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def version_of(package: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_script_version(script_name: str, script_path: Path) -> str:
    root = repo_root()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
        rel = script_path.resolve().relative_to(root)
        diff = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(rel)],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        suffix = "-dirty" if diff.returncode == 1 else ""
        return f"{script_name}@{commit}{suffix}"
    except Exception:
        return f"{script_name}@unknown"


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
