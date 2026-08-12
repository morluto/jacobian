"""Executable ownership checks for the built-in composition boundary."""

from __future__ import annotations

import ast
from pathlib import Path

from jacobian.portfolio.builtin import BUILTIN_PORTFOLIO_COMPONENT_FACTORIES

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


def test_only_explicit_builtin_composition_imports_every_domain_factory() -> None:
    factories = BUILTIN_PORTFOLIO_COMPONENT_FACTORIES
    assert factories, "expected explicit builtin domain factories"
    assert len(factories) == len(set(factories)), "duplicate domain factories"
    central_installers = (
        SOURCE / "portfolio" / "assembler.py",
        SOURCE / "portfolio" / "core_installation.py",
        SOURCE / "portfolio" / "domain_installation.py",
        SOURCE / "portfolio" / "foundation_installation.py",
    )
    assert all(
        not any(module.startswith("jacobian.domains.") for module in _imports(path))
        for path in central_installers
    )
