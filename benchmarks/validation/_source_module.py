"""Explicit path-bound loading for pure benchmark source-module tests."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import pytest


def load_source_module(
    module_name: str,
    path: Path,
    *,
    aliases: Mapping[str, ModuleType] | None = None,
) -> ModuleType:
    source = path.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"source module is not a regular file: {source}")
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ValueError(f"source module cannot be loaded: {source}")
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as module_state:
        module_state.setitem(sys.modules, module_name, module)
        for alias, aliased_module in (aliases or {}).items():
            module_state.setitem(sys.modules, alias, aliased_module)
        spec.loader.exec_module(module)
    return module


def load_task_verifier(task: Path, *, module_name: str) -> ModuleType:
    tests = task.resolve(strict=True) / "tests"
    support = load_source_module(
        f"{module_name}_support",
        tests / "verifier_support.py",
    )
    return load_source_module(
        module_name,
        tests / "verifier.py",
        aliases={"verifier_support": support},
    )


__all__ = ["load_source_module", "load_task_verifier"]
