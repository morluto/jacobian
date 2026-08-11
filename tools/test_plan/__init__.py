"""Authoritative test/CI plan compiler package."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CompileResult",
    "TestPlanManifest",
    "compile_manifest",
    "load_manifest",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from tools.test_plan import compile as compile_module

        return getattr(compile_module, name)
    raise AttributeError(name)
