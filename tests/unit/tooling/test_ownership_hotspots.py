"""Executable ownership boundaries for portfolio, providers, MCP, and Harbor."""

from __future__ import annotations

import ast
from pathlib import Path

from benchmarks.tooling.harbor_suite import load_registry

from jacobian.adapters.mcp.server import JacobianCoreExtension
from jacobian.domains.builtins import BUILTIN_DOMAIN_BUNDLE_FACTORIES

ROOT = Path(__file__).parents[3]
SOURCE = ROOT / "src" / "jacobian"


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text())
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return tuple(modules)


def test_only_explicit_builtin_composition_imports_every_domain_factory() -> None:
    factories = BUILTIN_DOMAIN_BUNDLE_FACTORIES
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


def test_shared_provider_mechanics_do_not_import_provider_implementations() -> None:
    imports = _imports(SOURCE / "provider_runtime.py")
    assert not any(module.startswith("jacobian.providers.") for module in imports)


def test_core_extension_exposes_exactly_the_stable_capability_tools() -> None:
    extension = JacobianCoreExtension(None, None)
    assert extension.identifier == "io.jacobian/core"
    assert extension.settings() == {"version": "1"}
    assert tuple(binding.kwargs["name"] for binding in extension.tools()) == (
        "capability.describe",
        "capability.invoke",
    )


def test_harbor_verifier_support_copies_are_identical() -> None:
    source = (ROOT / "benchmarks" / "tooling" / "verifier_support.py").read_bytes()
    targets = sorted(
        path
        for suite in load_registry()
        for path in suite.path.glob("*/tests/verifier_support.py")
    )
    expected = {
        ref.path / "tests" / "verifier_support.py"
        for suite in load_registry()
        for ref in suite.tasks
    }
    assert set(targets) == expected
    assert all(target.read_bytes() == source for target in targets)
