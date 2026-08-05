"""Executable checks for fixture ownership and state isolation."""

from __future__ import annotations

import ast
import runpy
from pathlib import Path

import pytest
from tests.support.state import copy_template, publish_template

ROOT = Path(__file__).parents[2]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_root_conftest_has_no_high_cost_imports_or_runtime_construction() -> None:
    """Collection of all tiers must not initialize the application runtime."""

    path = ROOT / "conftest.py"
    imports = _imports(path)
    assert not any(
        module.startswith(
            (
                "jacobian.runtime",
                "jacobian.portfolio",
                "jacobian.provider_runtime",
                "jacobian.domains",
                "sympy",
                "networkx",
                "sqlite3",
            )
        )
        for module in imports
    )


def test_complete_runtime_plugin_is_registered_only_by_owning_tiers() -> None:
    """Complete-runtime fixtures are registered once at the pytest root."""

    plugin_names = (
        "tests.support.runtime_templates",
        "tests.support.runtime_instances",
    )
    namespace = runpy.run_path(str(ROOT / "conftest.py"))
    assert namespace["pytest_plugins"] == plugin_names

    for tier in (
        "component",
        "domain",
        "composition",
        "e2e",
        "boundary" / Path("storage"),
        "boundary" / Path("providers"),
    ):
        imports = _imports(ROOT / tier / "conftest.py")
        assert not any(name in imports for name in plugin_names)
        assert not any(module.startswith("jacobian.portfolio") for module in imports)


def test_failed_template_build_has_no_reusable_partial_directory(
    tmp_path: Path,
) -> None:
    """A killed/failed builder cannot leave a target that looks complete."""

    target = tmp_path / "template"

    def fail(staging: Path) -> None:
        (staging / "partial.sqlite3").write_text("incomplete", encoding="utf-8")
        raise RuntimeError("simulated construction failure")

    with pytest.raises(RuntimeError, match="construction failure"):
        publish_template(target, fail)

    assert not target.exists()
    assert not list(tmp_path.glob(".template.staging-*"))
    assert not (tmp_path / "template.ready").exists()


def test_template_isolation_gives_each_test_mutable_state(tmp_path: Path) -> None:
    """Mutating one copied state directory cannot mutate the template."""

    template = tmp_path / "template"
    template.mkdir()
    (template / "metadata.txt").write_text("immutable", encoding="utf-8")
    first = copy_template(template, tmp_path / "first")
    second = copy_template(template, tmp_path / "second")

    (first / "metadata.txt").write_text("first mutation", encoding="utf-8")

    assert (template / "metadata.txt").read_text(encoding="utf-8") == "immutable"
    assert (second / "metadata.txt").read_text(encoding="utf-8") == "immutable"
