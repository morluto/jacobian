"""Executable ownership checks for the built-in composition boundary."""

from __future__ import annotations

import ast
from pathlib import Path

from jacobian.portfolio.builtin import BUILTIN_OPERATION_MODULES

ROOT = Path(__file__).parents[3]
SOURCE = ROOT / "src" / "jacobian"


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return tuple(modules)


def test_builtin_inventory_is_explicit_without_importing_domain_modules() -> None:
    modules = BUILTIN_OPERATION_MODULES
    assert modules, "expected explicit built-in operation modules"
    assert len(modules) == len(set(modules)), "duplicate operation modules"
    assert all(module.startswith("jacobian.domains.") for module, _factory in modules)
    central_installers = (
        SOURCE / "portfolio" / "assembler.py",
        SOURCE / "portfolio" / "foundation_binding.py",
    )
    assert all(
        not any(module.startswith("jacobian.domains.") for module in _imports(path))
        for path in central_installers
    )
    core_imports = _imports(SOURCE / "portfolio" / "core_binding.py")
    assert {
        module for module in core_imports if module.startswith("jacobian.domains.")
    } == {
        "jacobian.domains.polynomial_nullstellensatz.core",
        "jacobian.domains.polynomial_nullstellensatz.singular",
    }
