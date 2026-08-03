"""Executable ownership checks for shared provider mechanics."""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).parents[3] / "src" / "jacobian"


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return tuple(modules)


def test_shared_provider_mechanics_do_not_import_provider_implementations() -> None:
    imports = _imports(SOURCE / "provider_runtime.py")
    assert not any(module.startswith("jacobian.providers.") for module in imports)
