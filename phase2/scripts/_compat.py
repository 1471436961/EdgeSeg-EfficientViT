"""Import-time compatibility helpers for Phase 2 scripts.

These helpers are intentionally import-only. They let EfficientViT's upstream
import graph load on Windows without installing Triton or starting wandb side
effects. They must not change model math.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Dict, List


class _FakeSymbol:
    __slots__ = ()

    def __getattr__(self, name: str) -> "_FakeSymbol":
        return self

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "Fake Triton symbol was called. The Triton stub is import-only "
            "and must not execute kernels in Phase 2 export."
        )

    def __getitem__(self, item):
        raise RuntimeError(
            "Fake Triton symbol was indexed/launched. The Triton stub is "
            "import-only and must not execute kernels in Phase 2 export."
        )

    def __repr__(self) -> str:
        return "<FakeTritonSymbol import-only>"


_FAKE_TRITON_SYMBOL = _FakeSymbol()


def install_triton_stub() -> bool:
    """Install a no-op Triton module for upstream import compatibility.

    Call this AFTER importing torch and BEFORE importing efficientvit.*.
    Returns True if a stub was installed.
    """
    if "triton" in sys.modules:
        return False

    class _FakeTritonKernel:
        def __init__(self, fn) -> None:
            self.fn = fn
            self.__name__ = getattr(fn, "__name__", "fake_triton_kernel")
            self.__doc__ = getattr(fn, "__doc__", None)

        def __getitem__(self, grid):
            def _launch(*args, **kwargs):
                raise RuntimeError(
                    "Triton is unavailable. This stub only supports importing "
                    "EfficientViT modules; Triton kernels must not run for the "
                    "EfficientViT-Seg-B0 export path."
                )

            return _launch

    def _jit(fn=None, **kwargs):
        if fn is None:
            return lambda f: _FakeTritonKernel(f)
        return _FakeTritonKernel(fn)

    stub = types.ModuleType("triton")
    lang = types.ModuleType("triton.language")

    stub.jit = _jit
    stub.language = lang
    lang.constexpr = object  # type: ignore[attr-defined]

    stub.__file__ = None
    stub.__path__ = []
    stub.__loader__ = None
    stub.__package__ = "triton"
    stub.__spec__ = importlib.machinery.ModuleSpec("triton", loader=None)

    lang.__file__ = None
    lang.__path__ = []
    lang.__loader__ = None
    lang.__package__ = "triton.language"
    lang.__spec__ = importlib.machinery.ModuleSpec("triton.language", loader=None)

    def _stub_getattr(name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _FAKE_TRITON_SYMBOL

    stub.__getattr__ = _stub_getattr  # type: ignore[attr-defined]
    lang.__getattr__ = _stub_getattr  # type: ignore[attr-defined]

    sys.modules["triton"] = stub
    sys.modules["triton.language"] = lang
    return True


class _FakeWandbRun:
    def log(self, *args, **kwargs) -> None:
        return None

    def finish(self, *args, **kwargs) -> None:
        return None


def install_wandb_stub() -> bool:
    """Install a minimal import-only wandb module."""
    if "wandb" in sys.modules:
        return False

    stub = types.ModuleType("wandb")
    stub.__file__ = None
    stub.__loader__ = None
    stub.__package__ = "wandb"
    stub.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)

    def _init(*args, **kwargs) -> _FakeWandbRun:
        return _FakeWandbRun()

    def _noop(*args, **kwargs) -> None:
        return None

    stub.init = _init  # type: ignore[attr-defined]
    stub.login = _noop  # type: ignore[attr-defined]
    stub.log = _noop  # type: ignore[attr-defined]
    stub.finish = _noop  # type: ignore[attr-defined]
    sys.modules["wandb"] = stub
    return True


def install_import_compat_patches() -> Dict[str, object]:
    """Install all Phase 2 import-time patches and return metadata."""
    patches: List[str] = []
    triton_stubbed = install_triton_stub()
    wandb_stubbed = install_wandb_stub()
    if triton_stubbed:
        patches.append("triton_stub")
    if wandb_stubbed:
        patches.append("wandb_stub")
    return {
        "env_patches": patches,
        "triton_stubbed": triton_stubbed,
        "wandb_stubbed": wandb_stubbed,
    }
