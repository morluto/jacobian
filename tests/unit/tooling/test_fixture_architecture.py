"""Executable checks for fixture ownership and state isolation."""

from __future__ import annotations

import ast
import runpy
from functools import cache
from pathlib import Path

import pytest
from tests.support.state import copy_template, publish_template

ROOT = Path(__file__).parents[2]
REPOSITORY_ROOT = ROOT.parent


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _fixture_scope(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    for decorator in node.decorator_list:
        call = decorator if isinstance(decorator, ast.Call) else None
        target = call.func if call is not None else decorator
        is_fixture = (isinstance(target, ast.Name) and target.id == "fixture") or (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "pytest"
            and target.attr == "fixture"
        )
        if not is_fixture:
            continue
        if call is None:
            return "function"
        for keyword in call.keywords:
            if (
                keyword.arg == "scope"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                return keyword.value.value
        return "function"
    return None


@cache
def _fixture_functions() -> tuple[
    tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef], ...
]:
    fixtures: list[tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    roots = (ROOT, REPOSITORY_ROOT / "benchmarks" / "validation")
    for fixture_root in roots:
        for path in sorted(fixture_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            fixtures.extend(
                (path, node)
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and _fixture_scope(node) is not None
            )
    return tuple(fixtures)


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


def test_complete_runtime_fixtures_are_registered_only_by_owning_tiers() -> None:
    """Complete-runtime fixtures register under owning tiers, not the root."""

    full_fixture_names = (
        "complete_portfolio_template",
        "authorized_portfolio_template",
        "fresh_complete_runtime",
        "attached_complete_runtime",
        "attached_complete_runtime_read_only",
        "authorized_complete_runtime",
        "authorized_complete_runtime_read_only",
    )
    root = runpy.run_path(str(ROOT / "conftest.py"))
    assert root.get("pytest_plugins") == ("tests.support.resource_closure_plugin",)
    assert "pytest_plugins" not in root or "runtime" not in str(
        root.get("pytest_plugins")
    )

    full_owners = (
        ROOT / "composition" / "conftest.py",
        ROOT / "e2e" / "conftest.py",
        ROOT / "boundary" / "storage" / "conftest.py",
        ROOT / "boundary" / "providers" / "conftest.py",
    )
    for path in full_owners:
        namespace = runpy.run_path(str(path))
        assert "pytest_plugins" not in namespace
        for name in full_fixture_names:
            assert name in namespace, f"{path} missing {name}"

    mcp = runpy.run_path(str(ROOT / "boundary" / "mcp" / "conftest.py"))
    assert "pytest_plugins" not in mcp
    assert "attached_complete_runtime" in mcp
    assert "complete_portfolio_template" in mcp
    assert "fresh_complete_runtime" not in mcp
    assert "authorized_complete_runtime" not in mcp

    for tier in ("component", "domain", "unit"):
        confest = ROOT / tier / "conftest.py"
        if not confest.exists():
            continue
        imports = _imports(confest)
        assert "tests.support.complete_runtime_fixtures" not in imports
        assert "tests.support.runtime_templates" not in imports
        assert "tests.support.runtime_instances" not in imports
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


def test_broad_fixtures_do_not_share_mutable_test_services() -> None:
    """Service graphs and repositories stay local unless frozen as templates."""

    mutable_constructors = {
        "ArtifactRepository",
        "CapabilityService",
        "create_runtime",
        "open_domain_services",
    }
    violations: list[tuple[str, str, str | None, tuple[str, ...]]] = []
    for path, node in _fixture_functions():
        scope = _fixture_scope(node)
        if scope == "function":
            continue
        called_names = {
            name
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and (name := _call_name(call)) is not None
        }
        mutable_calls = tuple(sorted(called_names & mutable_constructors))
        if mutable_calls and "publish_template" not in called_names:
            violations.append(
                (
                    path.relative_to(REPOSITORY_ROOT).as_posix(),
                    node.name,
                    scope,
                    mutable_calls,
                )
            )

    assert violations == []


def test_resource_owning_fixtures_close_the_resource_they_construct() -> None:
    """A fixture-owned runtime or repository must have deterministic teardown."""

    resource_constructors = {"ArtifactRepository", "create_runtime"}
    leaks: list[tuple[str, str, str]] = []
    for path, node in _fixture_functions():
        owned: set[str] = set()
        for assignment in ast.walk(node):
            if isinstance(assignment, ast.Assign):
                value = assignment.value
                targets = assignment.targets
            elif isinstance(assignment, ast.AnnAssign):
                value = assignment.value
                targets = (assignment.target,)
            else:
                continue
            if (
                not isinstance(value, ast.Call)
                or _call_name(value) not in resource_constructors
            ):
                continue
            owned.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )
        closed = {
            call.func.value.id
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "close"
            and isinstance(call.func.value, ast.Name)
        }
        leaks.extend(
            (
                path.relative_to(REPOSITORY_ROOT).as_posix(),
                node.name,
                resource,
            )
            for resource in sorted(owned - closed)
        )

    assert leaks == []
