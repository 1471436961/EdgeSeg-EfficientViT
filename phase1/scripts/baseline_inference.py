"""
EdgeSeg-EfficientViT · Phase 1 · baseline_inference.py
=======================================================

Goal
----
Establish a reproducible, Nsight-friendly PyTorch baseline for
EfficientViT-Seg-B0 on a single MX250 GPU (Pascal sm_61, 2GB VRAM).

This script is the **dual-run NVTX baseline** described in
``phase1/design_notes/baseline_inference_design.md``:

* ``--nvtx-level A``  : no NVTX (clean latency reference)
* ``--nvtx-level B``  : mid-grain ranges (stem / stage1..4 / head)
* ``--nvtx-level C``  : LiteMLA-internal ranges via Plan-C
                       instance-level monkey-patch (for Plugin design)

Hard contracts (see design note for full rationale)
---------------------------------------------------
* batch_size = 1, dtype = fp32, CUDA Events timing, warmup=20, measure=100.
* Plan-C patch is **instance-level**, idempotent, restorable.
* sanity_check (Plan-C only) compares per-LiteMLA original vs patched
  forward on identical input, atol=rtol=1e-5.
* No torch.no_grad(); everything inference-bound runs under
  torch.inference_mode().
* Hashing / sanity / MACs all happen **outside** the measure loop.
* ``--weights`` is mandatory unless ``--allow-random-weights`` is set;
  the latter is for smoke tests only.

CLI
---
    python phase1/scripts/baseline_inference.py \
        --weights phase1/weights/b0.pt \
        --resolution 1024 2048 \
        --nvtx-level B \
        --measurement-mode latency \
        --warmup 20 --measure 100 \
        --out phase1/results/metrics/baseline_b0_levelB.json

See ``phase1/scripts/README.md`` for ready-to-run examples (incl. Nsight).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Environment compatibility patch: Triton stub (Windows / MX250)               #
# --------------------------------------------------------------------------- #
# Timing of installation (IMPORTANT -- read before moving this code):
#
#   The stub MUST be installed AFTER `import torch`, but BEFORE the first
#   `import efficientvit.*`. If we install it before importing torch, PyTorch
#   sees `sys.modules["triton"]` and incorrectly assumes Triton is available,
#   triggering its triton-aware registration paths (torch._dynamo,
#   torch.cuda._lazy_init, torch._library.custom_op, ...). That cascade then
#   pokes attributes on our stub during torch import itself, producing far
#   harder-to-debug failures than the original ModuleNotFoundError.
#
#   On a Triton-less Windows box, the correct ordering is:
#       1. import torch              (torch sees NO triton -> picks fallbacks)
#       2. _install_triton_stub()    (now sys.modules["triton"] becomes a stub)
#       3. import efficientvit       (upstream `import triton` resolves to stub)
#
#   This module therefore only DEFINES `_install_triton_stub()` here and keeps
#   `_TRITON_STUBBED = False` as a placeholder. The actual call happens inside
#   `build_model()` -- see the explicit comment marker there.
#
# Why it exists at all:
#   EfficientViT's upstream `models/nn/norm.py` unconditionally imports
#   `models/nn/triton_rms_norm.py`, which does:
#       import triton
#       import triton.language as tl
#       @triton.jit
#       def _rms_norm_2d_fwd_fused(..., BLOCK_SIZE: tl.constexpr): ...
#   B0 (per phase1/architecture_analysis.md conclusion #1) contains ZERO
#   LayerNorm / TritonRMSNorm modules, so the kernel is imported but never
#   launched. The stub covers exactly the IMPORT-TIME surface (triton.jit,
#   triton.language, triton.language.constexpr); any actual kernel launch is
#   intercepted by `_FakeTritonKernel.__getitem__` and raises a clear
#   RuntimeError -- a loud, attributable failure instead of silently wrong
#   numbers.
#
# Transparency: the JSON output records `env_patches=["triton_stub"]` and
# `triton_stubbed=true` so any downstream consumer can tell this run used the
# compatibility layer. See phase1/design_notes/baseline_inference_design.md
# section 11 for the full rationale (including why we use `_FakeSymbol`
# instead of returning `object` from the PEP-562 fallback).


class _FakeSymbol:
    """Sentinel returned by the triton-stub PEP 562 fallback `__getattr__`.

    Used purely for IMPORT-TIME attribute probing inside Torch / EfficientViT.
    Any actual call / subscript / launch attempt raises RuntimeError so that
    a kernel run on a Triton-less box fails loudly, with location info.

    Returning self from `__getattr__` lets chained access (e.g.
    ``triton.language.foo.bar``) keep resolving without surprising downstream
    code; returning the `object` class instead leads to bizarre AttributeError
    chains deep inside ``inspect.findsource`` / ``torch._library.custom_op``.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> "_FakeSymbol":
        return self

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "Fake Triton symbol was called. The Triton stub is import-only "
            "and must not execute kernels on this Windows B0 baseline path. "
            "See phase1/design_notes/baseline_inference_design.md section 11."
        )

    def __getitem__(self, item):
        raise RuntimeError(
            "Fake Triton symbol was indexed/launched. The Triton stub is "
            "import-only and must not execute kernels. "
            "See phase1/design_notes/baseline_inference_design.md section 11."
        )

    def __repr__(self) -> str:
        return "<FakeTritonSymbol import-only>"


_FAKE_TRITON_SYMBOL = _FakeSymbol()


def _install_triton_stub() -> bool:
    """Inject a no-op `triton` module so EfficientViT's unconditional
    triton import succeeds on Windows.

    MUST be called AFTER `import torch` and BEFORE any `import efficientvit.*`.
    See the module-level docstring above for the rationale.

    Returns True if a stub was installed, False if a real `triton` package
    was already present in `sys.modules` (in which case we do nothing).
    """
    if "triton" in sys.modules:
        return False

    class _FakeTritonKernel:
        """Looks like a `@triton.jit`-decorated kernel at import time,
        explodes loudly if anyone ever tries to launch it."""

        def __init__(self, fn) -> None:
            self.fn = fn
            self.__name__ = getattr(fn, "__name__", "fake_triton_kernel")
            self.__doc__ = getattr(fn, "__doc__", None)

        def __getitem__(self, grid):  # mirror real triton kernel launch API
            def _launch(*args, **kwargs):
                raise RuntimeError(
                    "Triton is unavailable in this Windows environment. "
                    "This stub exists only to import EfficientViT modules; "
                    "TritonRMSNorm must not be executed for "
                    "EfficientViT-Seg-B0 (see "
                    "phase1/architecture_analysis.md conclusion #1: B0 has "
                    "ZERO LayerNorm / TritonRMSNorm). If you see this error, "
                    "you are running a NON-B0 path on a Triton-less machine."
                )
            return _launch

    def _jit(fn=None, **kwargs):
        # Support both `@triton.jit` and `@triton.jit(...)` decorator forms.
        if fn is None:
            return lambda f: _FakeTritonKernel(f)
        return _FakeTritonKernel(fn)

    stub = types.ModuleType("triton")
    lang = types.ModuleType("triton.language")

    stub.jit = _jit
    stub.language = lang
    # `: tl.constexpr` annotations are evaluated at function definition time.
    # `object` is fine here because it's only used as an annotation token --
    # nothing ever instantiates it. The risky case (PyTorch dynamo poking
    # `triton.language.dtype` etc.) goes through `__getattr__` below, which
    # returns `_FAKE_TRITON_SYMBOL` instead of `object`.
    lang.constexpr = object  # type: ignore[attr-defined]

    # ----- Module metadata dunders (MUST be set explicitly) -----------------
    # PEP 562 module `__getattr__` is only consulted when normal attribute
    # lookup fails. CPython's `inspect.getfile(module)` does:
    #     if getattr(object, '__file__', None): return object.__file__
    # Without an explicit `__file__`, `getattr(stub, '__file__', None)` falls
    # through to our PEP 562 fallback, which returns `_FAKE_TRITON_SYMBOL`
    # (truthy!). Inspect then treats the fake symbol as a filename string,
    # calls `.endswith(...)`, which triggers `_FakeSymbol.__call__` and blows
    # up with the import-only RuntimeError -- deep inside torchvision's
    # `_meta_registrations` import path.
    #
    # Fix: pre-populate every module-metadata dunder that any std-lib code
    # could read at import time, with safe falsy values, BEFORE installing
    # the PEP 562 catch-all. inspect / importlib will see "this is a builtin-
    # like module with no source file" and gracefully return None.
    import importlib.machinery as _mach
    stub.__file__ = None
    stub.__path__ = []          # mark as a (namespace) package
    stub.__loader__ = None
    stub.__package__ = "triton"
    stub.__spec__ = _mach.ModuleSpec("triton", loader=None)
    lang.__file__ = None
    lang.__path__ = []
    lang.__loader__ = None
    lang.__package__ = "triton.language"
    lang.__spec__ = _mach.ModuleSpec("triton.language", loader=None)

    # ----- PEP 562 catch-all (installed AFTER the dunders above) ------------
    # PyTorch and EfficientViT probe additional triton.language symbols at
    # import time. Rather than enumerate every one, install a PEP 562
    # module-level `__getattr__` that returns `_FAKE_TRITON_SYMBOL` for
    # anything we haven't explicitly defined.
    #
    # Returning the singleton (rather than the `object` type) avoids
    # surprising downstream code paths like `inspect.getsourcefile` ->
    # `filename.endswith(...)`, which assumes string-like duck-typing and
    # crashes on receiving a bare type object.
    #
    # We additionally raise AttributeError for any dunder name we did not
    # pre-populate above -- otherwise std-lib introspection (inspect /
    # importlib / pickle / copy) starts pulling fake symbols into places
    # where it expects real strings, None, or AttributeError.
    _PROBED_DUNDERS_ALLOWED = {
        "__file__", "__path__", "__loader__", "__package__",
        "__spec__", "__name__", "__doc__",
    }

    def _stub_getattr(name: str):
        if name.startswith("__") and name.endswith("__"):
            # All dunders we care about are pre-populated above. Anything
            # else (e.g. inspect probing `__qualname__` on a module) should
            # surface as a clean AttributeError, NOT a FakeSymbol.
            raise AttributeError(name)
        return _FAKE_TRITON_SYMBOL
    stub.__getattr__ = _stub_getattr  # type: ignore[attr-defined]
    lang.__getattr__ = _stub_getattr  # type: ignore[attr-defined]

    sys.modules["triton"] = stub
    sys.modules["triton.language"] = lang
    return True


# Placeholder. Real value is set inside `build_model()` immediately before the
# first `import efficientvit.*`. Anything that reads this before `build_model`
# runs should treat it as "not yet installed".
_TRITON_STUBBED: bool = False


# NOTE: `import torch` happens here WITHOUT any triton stub in `sys.modules`.
# See the timing comment block above for why this ordering matters.
import numpy as np
import torch

# --------------------------------------------------------------------------- #
# 0. Constants                                                                 #
# --------------------------------------------------------------------------- #

JSON_SCHEMA_VERSION = "1.0"
SCRIPT_NAME = "baseline_inference.py"

# Plan-C sanity check tolerance (FP32). Loosen for FP16/AMP in the future.
SANITY_ATOL = 1e-5
SANITY_RTOL = 1e-5

# Sentinel attribute names set on patched LiteMLA instances.
_PATCH_FLAG = "_edgeseg_nvtx_patched"
_PATCH_ORIG = "_edgeseg_original_forward"


# --------------------------------------------------------------------------- #
# 1. Argument parsing                                                          #
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EfficientViT-Seg-B0 baseline latency benchmark (Phase 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Model / weights ---------------------------------------------------- #
    p.add_argument("--model", default="b0",
                   choices=["b0"],
                   help="EfficientViT-Seg variant. Phase 1 only supports b0.")
    p.add_argument("--dataset", default="cityscapes",
                   choices=["cityscapes"],
                   help="Pre-training dataset (only affects head config / # classes).")
    p.add_argument("--weights", type=str, default=None,
                   help="Path to .pt checkpoint. Required unless "
                        "--allow-random-weights is set.")
    p.add_argument("--allow-random-weights", action="store_true",
                   help="Smoke-test only. Use randomly initialized weights. "
                        "JSON will be marked is_smoke_test=true.")

    # --- Input / device ----------------------------------------------------- #
    p.add_argument("--resolution", type=int, nargs=2, default=[1024, 2048],
                   metavar=("H", "W"),
                   help="Input spatial size (H W). Cityscapes default 1024x2048.")
    p.add_argument("--input-image", type=str, default=None,
                   help="Optional fixed input image path (preprocessed to tensor). "
                        "If omitted, a fixed-seed dummy tensor is used.")
    p.add_argument("--device", default="cuda",
                   help="Torch device. Phase 1 expects 'cuda'.")
    p.add_argument("--seed", type=int, default=2026,
                   help="Seed for dummy input + any RNG consumers.")

    # --- NVTX / profiling --------------------------------------------------- #
    p.add_argument("--nvtx-level", default="A",
                   choices=["A", "B", "C"],
                   help="A=no NVTX, B=mid-grain, C=LiteMLA-internal (Plan C).")
    p.add_argument("--profile-macs", action="store_true",
                   help="Optional FLOPs/MACs profiling via torchprofile. "
                        "Runs once, outside measure loop. May increase VRAM.")

    # --- Timing ------------------------------------------------------------- #
    p.add_argument("--warmup", type=int, default=20,
                   help="Warmup iterations (not measured).")
    p.add_argument("--measure", type=int, default=100,
                   help="Measured iterations.")
    p.add_argument("--measurement-mode", default="latency",
                   choices=["latency", "throughput"],
                   help="latency: per-iter CUDA Event timing -> p50/p95/p99. "
                        "throughput: batch-enqueue then single sync -> FPS.")
    p.add_argument("--cudnn-benchmark", default="on",
                   choices=["on", "off"],
                   help="torch.backends.cudnn.benchmark. Default on for fixed shape.")

    # --- Output ------------------------------------------------------------- #
    p.add_argument("--out", type=str, required=False, default=None,
                   help="Output JSON path. If omitted, a name is auto-derived "
                        "under phase1/results/metrics/.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build model + parse args + print intended JSON path, "
                        "then exit without timing. Used by CI / linting.")
    return p


def validate_args(args: argparse.Namespace) -> None:
    """Enforce contracts that argparse cannot express on its own."""
    # Contract #7: missing weights is a hard error unless explicitly opted out.
    if not args.weights and not args.allow_random_weights:
        raise SystemExit(
            "ERROR: --weights is required for an official baseline run. "
            "Pass --allow-random-weights only for smoke tests."
        )
    if args.weights and args.allow_random_weights:
        raise SystemExit(
            "ERROR: --weights and --allow-random-weights are mutually exclusive."
        )
    if args.warmup < 0 or args.measure <= 0:
        raise SystemExit("ERROR: --warmup must be >=0 and --measure must be >0.")
    if args.nvtx_level == "C" and args.device != "cuda":
        raise SystemExit("ERROR: --nvtx-level=C requires CUDA device.")


# --------------------------------------------------------------------------- #
# 2. Environment / metadata helpers                                            #
# --------------------------------------------------------------------------- #

def _run_git(args: List[str], cwd: Path) -> Optional[str]:
    """Run a git command; return stripped stdout or None on failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def resolve_script_version() -> str:
    """
    Return e.g. ``baseline_inference.py@a1b2c3d`` (clean) or
    ``baseline_inference.py@a1b2c3d-dirty`` (uncommitted) or
    ``baseline_inference.py@uncommitted`` / ``baseline_inference.py@git_unavailable``.

    Implementation per user constraint #3:
      1. ``git rev-parse --show-toplevel`` -> repo_root
      2. compute __file__ relative to repo_root
      3. ``git diff --quiet HEAD -- <relative_path>`` for dirty check
    """
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    repo_root_str = _run_git(["rev-parse", "--show-toplevel"], script_dir)
    if repo_root_str is None:
        return f"{SCRIPT_NAME}@git_unavailable"
    repo_root = Path(repo_root_str).resolve()

    commit = _run_git(["rev-parse", "--short=7", "HEAD"], repo_root)
    if commit is None:
        return f"{SCRIPT_NAME}@uncommitted"

    try:
        rel = script_path.relative_to(repo_root).as_posix()
    except ValueError:
        # Script lives outside the repo somehow. Be conservative.
        return f"{SCRIPT_NAME}@{commit}"

    diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", rel],
        cwd=str(repo_root),
        capture_output=True,
        check=False,
        timeout=5,
    )
    if diff.returncode == 0:
        return f"{SCRIPT_NAME}@{commit}"
    if diff.returncode == 1:
        return f"{SCRIPT_NAME}@{commit}-dirty"
    return f"{SCRIPT_NAME}@{commit}-unknown"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_tensor(t: torch.Tensor) -> str:
    """Hash a tensor's raw bytes deterministically (CPU, contiguous)."""
    arr = t.detach().to("cpu").contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def collect_env_meta() -> Dict[str, Any]:
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else None
    cudnn_ver = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    dev_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    return {
        "device_name": dev_name,
        "device_capability": f"sm_{dev_cap[0]}{dev_cap[1]}" if dev_cap else None,
        "torch_version": torch.__version__,
        "cuda_version": cuda_ver,
        "cudnn_version": cudnn_ver,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }


# --------------------------------------------------------------------------- #
# 3. Model construction                                                        #
# --------------------------------------------------------------------------- #

def build_model(args: argparse.Namespace) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Build EfficientViT-Seg-B0 and return (model, weights_meta).

    weights_meta contains:
        weights_status   : "loaded" | "random"
        weights_path     : str | None
        weights_sha256   : str | None
        weights_load_msg : str (missing/unexpected key summary)
    """
    # Make repo importable: phase1/scripts/baseline_inference.py
    #   -> repo root is parents[2] in *this* checkout layout
    #      (E:/.../EdgeSeg-EfficientViT/EdgeSeg-EfficientViT)
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # === Triton stub injection ===========================================
    # CRITICAL: this MUST happen here -- AFTER `import torch` (which already
    # ran at module-load time) and BEFORE the first `import efficientvit.*`.
    # See the module-level comment block on _install_triton_stub() for the
    # full rationale and design_notes/baseline_inference_design.md section 11.
    global _TRITON_STUBBED
    _TRITON_STUBBED = _install_triton_stub()
    # =====================================================================

    # Import after sys.path is patched and triton stub is in place.
    from efficientvit.seg_model_zoo import create_efficientvit_seg_model  # type: ignore

    # Build the zoo key. Upstream registers models under keys of the form
    # "efficientvit-seg-{variant}-{dataset}" (see seg_model_zoo.py).
    zoo_key = f"efficientvit-seg-{args.model}-{args.dataset}"

    weights_meta: Dict[str, Any] = {
        "weights_status": None,
        "weights_path": None,
        "weights_sha256": None,
        "weights_load_msg": None,
        "zoo_key": zoo_key,
    }

    if args.weights:
        wpath = Path(args.weights).expanduser().resolve()
        if not wpath.is_file():
            raise SystemExit(f"ERROR: --weights file not found: {wpath}")
        # pretrained=True forces create_efficientvit_seg_model to call
        # load_state_dict_from_file(weight_url) -- which is exactly what we
        # want, because we are passing an explicit checkpoint path.
        model = create_efficientvit_seg_model(
            name=zoo_key,
            pretrained=True,
            weight_url=str(wpath),
        )
        weights_meta.update({
            "weights_status": "loaded",
            "weights_path": str(wpath),
            "weights_sha256": sha256_of_file(wpath),
            "weights_load_msg": (
                f"create_efficientvit_seg_model(name={zoo_key!r}, "
                f"pretrained=True) loaded checkpoint"
            ),
        })
    else:
        # --allow-random-weights branch: smoke-test only. pretrained=False
        # builds the architecture but skips load_state_dict_from_file.
        model = create_efficientvit_seg_model(
            name=zoo_key,
            pretrained=False,
            weight_url=None,
        )
        weights_meta.update({
            "weights_status": "random",
            "weights_path": None,
            "weights_sha256": None,
            "weights_load_msg": (
                f"create_efficientvit_seg_model(name={zoo_key!r}, "
                f"pretrained=False) random init (smoke test)"
            ),
        })

    model.eval().to(args.device)
    return model, weights_meta


def build_input_tensor(args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Build a fixed input tensor (NCHW, fp32) plus metadata.

    input_status = "image" | "dummy"
    """
    H, W = args.resolution
    if args.input_image:
        # Phase 1 we keep this minimal: load via PIL, ToTensor, normalize is
        # NOT applied (baseline forward is shape-only; mIoU is not measured here).
        from PIL import Image  # local import to avoid hard dep when unused
        img_path = Path(args.input_image).expanduser().resolve()
        if not img_path.is_file():
            raise SystemExit(f"ERROR: --input-image not found: {img_path}")
        img = Image.open(img_path).convert("RGB").resize((W, H))
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0).contiguous()
        meta = {
            "input_status": "image",
            "input_path": str(img_path),
            "input_sha256": sha256_of_file(img_path),
        }
    else:
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        t = torch.randn(1, 3, H, W, generator=g, dtype=torch.float32)
        meta = {
            "input_status": "dummy",
            "input_path": None,
            "input_sha256": sha256_of_tensor(t),
        }
    t = t.to(args.device, non_blocking=True)
    meta.update({"resolution": [H, W], "batch_size": 1, "dtype": "fp32"})
    return t, meta


# --------------------------------------------------------------------------- #
# 4. NVTX injection                                                            #
# --------------------------------------------------------------------------- #

def _nvtx_push(name: str) -> None:
    torch.cuda.nvtx.range_push(name)


def _nvtx_pop() -> None:
    torch.cuda.nvtx.range_pop()


# ---- Plan B: mid-grain hooks (stem / stage1..4 / head) -------------------- #

def _find_seg_components(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """
    Locate the named blocks we want Plan-B ranges around.

    The repo's EfficientViTSeg exposes:
        model.backbone (EfficientViTBackbone) -> .input_stem, .stages
        model.head     (SegHead)
    See phase1/architecture_analysis.md for layout.
    """
    out: Dict[str, torch.nn.Module] = {}
    bb = getattr(model, "backbone", None)
    if bb is not None:
        if hasattr(bb, "input_stem"):
            out["stem"] = bb.input_stem
        # `stages` is a nn.ModuleList of length 5 in upstream repo (stage0..4)
        stages = getattr(bb, "stages", None)
        if stages is not None:
            for i, s in enumerate(stages):
                out[f"stage{i}"] = s
    head = getattr(model, "head", None)
    if head is not None:
        out["head"] = head
    return out


def apply_plan_b_hooks(model: torch.nn.Module) -> List[Any]:
    """Register pre/forward hooks for mid-grain NVTX ranges. Returns handles."""
    handles: List[Any] = []
    comps = _find_seg_components(model)

    def _mk_pre(name: str) -> Callable[..., None]:
        def _pre(_mod, _inp):
            _nvtx_push(name)
        return _pre

    def _mk_post(_name: str) -> Callable[..., None]:
        def _post(_mod, _inp, _out):
            _nvtx_pop()
        return _post

    for name, mod in comps.items():
        handles.append(mod.register_forward_pre_hook(_mk_pre(name)))
        handles.append(mod.register_forward_hook(_mk_post(name)))
    return handles


def remove_hooks(handles: List[Any]) -> None:
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


# ---- Plan C: LiteMLA-internal instance-level monkey-patch ------------------ #

def _find_litemla_modules(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    """Return [(qualified_name, module), ...] of all LiteMLA instances."""
    from efficientvit.models.nn.ops import LiteMLA  # type: ignore
    found: List[Tuple[str, torch.nn.Module]] = []
    for n, m in model.named_modules():
        if isinstance(m, LiteMLA):
            found.append((n, m))
    return found


def _make_patched_litemla_forward(original_forward: Callable) -> Callable:
    """
    Wrap an unbound LiteMLA.forward with NVTX ranges that mirror the four
    structural stages we want to attribute in Nsight:

        litemla/qkv_proj  litemla/multiscale  litemla/relu_lin_attn  litemla/proj

    NOTE: this is a lightweight wrapper around the *whole* forward; we do not
    rewrite internal logic. Fine-grained internal NVTX (one range per Conv/MM)
    is intentionally NOT done here -- see design note for rationale.
    """
    def patched_forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[no-untyped-def]
        _nvtx_push("LiteMLA")
        try:
            out = original_forward(self, x)
        finally:
            _nvtx_pop()
        return out
    return patched_forward


def apply_plan_c_monkey_patch(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    """
    Per user constraint #1+#2: instance-level patch, idempotent, restorable.

    Returns the list of patched (name, module) pairs for later restore.
    """
    patched: List[Tuple[str, torch.nn.Module]] = []
    for name, m in _find_litemla_modules(model):
        if getattr(m, _PATCH_FLAG, False):
            # Already patched; skip (per constraint #1: avoid double-wrap).
            continue
        # Capture the *class-level* original forward as an unbound function.
        # We deliberately bind the original via the type, so even if some other
        # instance gets patched, this closure stays correct.
        original_forward = type(m).forward
        setattr(m, _PATCH_ORIG, original_forward)
        m.forward = types.MethodType(  # type: ignore[method-assign]
            _make_patched_litemla_forward(original_forward), m
        )
        setattr(m, _PATCH_FLAG, True)
        patched.append((name, m))
    return patched


def restore_plan_c_monkey_patch(patched: List[Tuple[str, torch.nn.Module]]) -> None:
    """Restore each instance's forward. Safe to call multiple times."""
    for _name, m in patched:
        if not getattr(m, _PATCH_FLAG, False):
            continue
        # Removing the instance attribute lets Python fall back to the class
        # method, which is the original unbound `LiteMLA.forward`.
        try:
            del m.forward  # type: ignore[attr-defined]
        except AttributeError:
            pass
        try:
            delattr(m, _PATCH_ORIG)
        except AttributeError:
            pass
        setattr(m, _PATCH_FLAG, False)


# --------------------------------------------------------------------------- #
# 5. Sanity check (Plan-C only)                                                #
# --------------------------------------------------------------------------- #

@dataclass
class SanityResult:
    passed: bool
    per_module: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""


def run_sanity_check(model: torch.nn.Module, x: torch.Tensor) -> SanityResult:
    """
    Per-LiteMLA original-vs-patched equivalence check (user constraint #4):

      1. forward-hook every LiteMLA, capture inp[0] and out (cloned).
      2. one forward pass with ORIGINAL model.
      3. remove hooks.
      4. apply Plan-C monkey-patch.
      5. for each LiteMLA, call patched forward on cached input directly.
      6. compare; require torch.allclose(atol=rtol=1e-5).

    On success: returns SanityResult(passed=True, per_module=[...]).
    On failure: returns SanityResult(passed=False, ...). Caller decides exit.
    """
    captured: Dict[str, Dict[str, torch.Tensor]] = {}
    litemlas = _find_litemla_modules(model)

    def _mk_hook(name: str) -> Callable[..., None]:
        def _hook(_mod, inp, out):
            # constraint #4: only forward-hook, detach+clone.
            captured[name] = {
                "inp": inp[0].detach().clone(),
                "out": out.detach().clone(),
            }
        return _hook

    handles = [m.register_forward_hook(_mk_hook(n)) for n, m in litemlas]
    try:
        with torch.inference_mode():
            _ = model(x)
    finally:
        remove_hooks(handles)

    # Now patch and re-run each LiteMLA in isolation.
    patched = apply_plan_c_monkey_patch(model)
    per_module: List[Dict[str, Any]] = []
    passed = True
    try:
        with torch.inference_mode():
            for name, m in litemlas:
                cap = captured.get(name)
                if cap is None:
                    per_module.append({"name": name, "ok": False,
                                       "reason": "no captured input"})
                    passed = False
                    continue
                y_patched = m(cap["inp"])  # uses patched forward
                y_orig = cap["out"]
                if y_patched.shape != y_orig.shape:
                    per_module.append({"name": name, "ok": False,
                                       "reason": f"shape mismatch "
                                                  f"{tuple(y_patched.shape)} vs "
                                                  f"{tuple(y_orig.shape)}"})
                    passed = False
                    continue
                diff = (y_patched - y_orig).abs()
                max_diff = float(diff.max().item())
                mean_diff = float(diff.mean().item())
                ok = bool(torch.allclose(y_patched, y_orig,
                                         atol=SANITY_ATOL, rtol=SANITY_RTOL))
                per_module.append({
                    "name": name,
                    "ok": ok,
                    "max_abs_diff": max_diff,
                    "mean_abs_diff": mean_diff,
                    "atol": SANITY_ATOL,
                    "rtol": SANITY_RTOL,
                })
                passed = passed and ok
    finally:
        # Keep the patch in place if passed -- caller will run measurement
        # with it. If failed, restore to leave the model in a clean state.
        if not passed:
            restore_plan_c_monkey_patch(patched)
    return SanityResult(passed=passed, per_module=per_module,
                        notes=f"checked {len(litemlas)} LiteMLA modules")


# --------------------------------------------------------------------------- #
# 6. MACs profiling (optional)                                                 #
# --------------------------------------------------------------------------- #

def maybe_profile_macs(model: torch.nn.Module, x: torch.Tensor) -> Dict[str, Any]:
    """
    Per user constraint #6: runs once, outside warmup/measure.

    Best-effort using torchprofile (already in requirements). On failure,
    we emit a warning and return status='unavailable' without raising.
    """
    info: Dict[str, Any] = {"status": "unavailable", "macs": None, "tool": None,
                            "error": None}
    try:
        from torchprofile import profile_macs  # type: ignore
        with torch.inference_mode():
            macs = profile_macs(model, (x,))
        info.update({"status": "ok", "macs": int(macs), "tool": "torchprofile"})
    except Exception as e:  # noqa: BLE001 - best-effort, MUST NOT abort timing
        info.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
    return info


# --------------------------------------------------------------------------- #
# 7. Timing                                                                    #
# --------------------------------------------------------------------------- #

def _percentiles(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std":  float(a.std(ddof=0)),
        "min":  float(a.min()),
        "max":  float(a.max()),
        "p50":  float(np.percentile(a, 50)),
        "p95":  float(np.percentile(a, 95)),
        "p99":  float(np.percentile(a, 99)),
    }


def measure_latency_per_iter(model: torch.nn.Module,
                             x: torch.Tensor,
                             warmup: int,
                             measure: int) -> Dict[str, Any]:
    """Per-iteration CUDA Event timing -> p50/p95/p99 (ms). Primary mode."""
    torch.cuda.synchronize()
    with torch.inference_mode():
        # Warmup
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()

        # Measure
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
        times_ms: List[float] = []
        for i in range(measure):
            starts[i].record()
            _ = model(x)
            ends[i].record()
            ends[i].synchronize()
            times_ms.append(starts[i].elapsed_time(ends[i]))
    stats = _percentiles(times_ms)
    return {"mode": "latency", "ms": stats, "samples": times_ms}


def measure_throughput_batched(model: torch.nn.Module,
                               x: torch.Tensor,
                               warmup: int,
                               measure: int) -> Dict[str, Any]:
    """Batch-enqueue then one sync -> FPS / aggregate ms. Secondary mode."""
    torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(measure):
            _ = model(x)
        end.record()
        end.synchronize()
        total_ms = start.elapsed_time(end)
    avg_ms = total_ms / max(1, measure)
    fps = 1000.0 / avg_ms if avg_ms > 0 else float("nan")
    return {
        "mode": "throughput",
        "total_ms": total_ms,
        "avg_ms": avg_ms,
        "fps": fps,
        "iterations": measure,
    }


# --------------------------------------------------------------------------- #
# 8. Memory                                                                    #
# --------------------------------------------------------------------------- #

def collect_memory_stats() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"max_memory_allocated_mb": None, "max_memory_reserved_mb": None}
    return {
        "max_memory_allocated_mb":
            torch.cuda.max_memory_allocated() / (1024 ** 2),
        "max_memory_reserved_mb":
            torch.cuda.max_memory_reserved() / (1024 ** 2),
    }


# --------------------------------------------------------------------------- #
# 9. Output                                                                    #
# --------------------------------------------------------------------------- #

def derive_default_out_path(args: argparse.Namespace) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    base = repo_root / "phase1" / "results" / "metrics"
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    H, W = args.resolution
    name = (f"baseline_{args.model}_{args.dataset}_{H}x{W}_"
            f"level{args.nvtx_level}_{args.measurement_mode}_{stamp}.json")
    return base / name


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)


# --------------------------------------------------------------------------- #
# 10. Main                                                                     #
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)

    # Reproducibility knobs (constraint #5: inference_mode used downstream).
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("ERROR: CUDA requested but torch.cuda.is_available()==False.")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = (args.cudnn_benchmark == "on")
        torch.backends.cudnn.deterministic = False
        torch.cuda.reset_peak_memory_stats()

    # ----- Build model + input (constraint #6: hashing happens here, NOT in timing) #
    model, weights_meta = build_model(args)
    x, input_meta = build_input_tensor(args)

    # ----- Optional MACs profiling, BEFORE NVTX/sanity (constraint #6) ------- #
    macs_info: Dict[str, Any] = {"status": "skipped"}
    if args.profile_macs:
        macs_info = maybe_profile_macs(model, x)

    # ----- NVTX injection + (Plan-C only) sanity check ----------------------- #
    nvtx_meta: Dict[str, Any] = {"level": args.nvtx_level, "applied": False,
                                  "patched_modules": [], "hook_count": 0}
    sanity_payload: Dict[str, Any] = {"performed": False, "passed": None,
                                       "per_module": [], "notes": ""}
    hook_handles: List[Any] = []
    patched_pairs: List[Tuple[str, torch.nn.Module]] = []

    try:
        if args.nvtx_level == "B":
            hook_handles = apply_plan_b_hooks(model)
            nvtx_meta.update({"applied": True, "hook_count": len(hook_handles)})
        elif args.nvtx_level == "C":
            sres = run_sanity_check(model, x)
            sanity_payload = {
                "performed": True,
                "passed": sres.passed,
                "per_module": sres.per_module,
                "notes": sres.notes,
                "atol": SANITY_ATOL,
                "rtol": SANITY_RTOL,
            }
            if not sres.passed:
                # Build a minimal failure JSON for forensics (constraint #7 does
                # NOT apply here -- this is not a 'missing weights' situation;
                # it's a numerical regression we MUST persist for diagnosis).
                fail_out = args.out and Path(args.out) or derive_default_out_path(args)
                fail_payload = _assemble_payload(
                    args, weights_meta, input_meta, nvtx_meta, sanity_payload,
                    macs_info, timing=None, memory=None,
                    status="sanity_failed",
                )
                save_json(fail_out, fail_payload)
                print(f"[FATAL] Plan-C sanity check failed. Report -> {fail_out}",
                      file=sys.stderr)
                return 3
            patched_pairs = [(n, m) for n, m in _find_litemla_modules(model)
                             if getattr(m, _PATCH_FLAG, False)]
            nvtx_meta.update({"applied": True,
                              "patched_modules": [n for n, _ in patched_pairs]})

        if args.dry_run:
            print("[dry-run] args validated, model/input built, NVTX prepared.")
            print(f"[dry-run] intended output: "
                  f"{args.out or derive_default_out_path(args)}")
            return 0

        # ----- Warmup + measure ---------------------------------------------- #
        torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
        if args.measurement_mode == "latency":
            timing = measure_latency_per_iter(model, x, args.warmup, args.measure)
        else:
            timing = measure_throughput_batched(model, x, args.warmup, args.measure)
        memory = collect_memory_stats()

    finally:
        # Tidy up: always restore.
        remove_hooks(hook_handles)
        if patched_pairs:
            restore_plan_c_monkey_patch(patched_pairs)

    # ----- Assemble + save JSON --------------------------------------------- #
    out_path = Path(args.out) if args.out else derive_default_out_path(args)
    payload = _assemble_payload(
        args, weights_meta, input_meta, nvtx_meta, sanity_payload,
        macs_info, timing=timing, memory=memory, status="ok",
    )
    save_json(out_path, payload)
    print(f"[OK] saved -> {out_path}")

    # Console summary for humans
    if args.measurement_mode == "latency":
        ms = timing["ms"]
        print(f"  latency (ms):  mean={ms['mean']:.3f}  p50={ms['p50']:.3f}  "
              f"p95={ms['p95']:.3f}  p99={ms['p99']:.3f}")
    else:
        print(f"  throughput:    avg={timing['avg_ms']:.3f} ms  "
              f"fps={timing['fps']:.2f}")
    return 0


def _assemble_payload(args: argparse.Namespace,
                      weights_meta: Dict[str, Any],
                      input_meta: Dict[str, Any],
                      nvtx_meta: Dict[str, Any],
                      sanity_payload: Dict[str, Any],
                      macs_info: Dict[str, Any],
                      timing: Optional[Dict[str, Any]],
                      memory: Optional[Dict[str, Any]],
                      status: str) -> Dict[str, Any]:
    env = collect_env_meta()
    return {
        "json_schema_version": JSON_SCHEMA_VERSION,
        "status": status,
        "script_version": resolve_script_version(),
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "is_smoke_test": bool(args.allow_random_weights),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "model": {
            "name": args.model,
            "dataset": args.dataset,
        },
        "weights": weights_meta,
        "input": input_meta,
        "env": {
            **env,
            "env_patches": ["triton_stub"] if _TRITON_STUBBED else [],
            "triton_stubbed": bool(_TRITON_STUBBED),
        },
        "cudnn": {
            "benchmark": torch.backends.cudnn.benchmark,
            "deterministic": torch.backends.cudnn.deterministic,
        },
        "nvtx": nvtx_meta,
        "sanity_check": sanity_payload,
        "macs": macs_info,
        "timing": timing,
        "memory": memory,
    }


if __name__ == "__main__":
    sys.exit(main())
