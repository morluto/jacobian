"""Tests for local execution of the checked-in pull-request test plan."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from tools.command_runner import ToolCommandResult, ToolCommandStatus

ROOT = Path(__file__).parents[2]


def _load() -> ModuleType:
    path = ROOT / "tools" / "affected_validation.py"
    spec = importlib.util.spec_from_file_location("affected_validation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_public_math_contract_selects_scoped_static_and_catalog_evidence() -> None:
    runner = _load()
    plan = runner.build_plan(
        event="pull_request",
        base_revision="a" * 40,
        head_revision="b" * 40,
        changed_paths=["src/jacobian/math/code_theory/_models.py"],
        repository=ROOT,
    )

    commands = runner.commands_for_plan(
        plan,
        paths=["src/jacobian/math/code_theory/_models.py"],
        repository=ROOT,
    )

    assert commands == (
        (
            "make",
            "lint-scoped",
            "PATHS=src/jacobian/math/code_theory/_models.py",
        ),
        (
            "make",
            "typecheck-scoped",
            "PATHS=src/jacobian/math/code_theory/_models.py",
        ),
        ("make", "test-math", "TESTS=tests/math/code_theory"),
        ("make", "test-catalog"),
        ("make", "test-integration", "TESTS=tests/integration/catalog/"),
    )


def test_runtime_boundary_selection_preserves_full_math_fallback() -> None:
    runner = _load()
    plan = runner.build_plan(
        event="pull_request",
        base_revision="a" * 40,
        head_revision="b" * 40,
        changed_paths=["src/jacobian/mcp/tools.py"],
        repository=ROOT,
    )

    commands = runner.commands_for_plan(
        plan,
        paths=["src/jacobian/mcp/tools.py"],
        repository=ROOT,
    )

    assert ("make", "test-math") in commands
    assert ("make", "test-mcp") in commands
    assert ("make", "test-process") in commands


def test_deleted_python_path_is_not_offered_to_scoped_static_tools() -> None:
    runner = _load()
    plan = runner.build_plan(
        event="pull_request",
        base_revision="a" * 40,
        head_revision="b" * 40,
        changed_paths=["tests/math/removed_test.py"],
        repository=ROOT,
    )

    commands = runner.commands_for_plan(
        plan,
        paths=["tests/math/removed_test.py"],
        repository=ROOT,
    )

    assert all(
        command[1] not in {"lint-scoped", "typecheck-scoped"} for command in commands
    )
    assert commands == (("make", "test-math"),)


def test_changed_paths_include_worktree_and_untracked_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load()

    def git(*arguments: str, repository: Path) -> str:
        del repository
        if arguments == ("rev-parse", "--verify", "origin/main"):
            return "a" * 40
        if arguments == ("rev-parse", "--verify", "HEAD"):
            return "b" * 40
        if arguments[:3] == ("diff", "--no-ext-diff", "--no-textconv"):
            return "src/jacobian/math/graphs/_operations.py\n"
        if arguments == ("diff", "--name-only"):
            return "tests/math/graphs/test_graph_coloring.py\n"
        if arguments == ("diff", "--cached", "--name-only"):
            return "src/jacobian/math/graphs/values.py\n"
        if arguments == ("ls-files", "--others", "--exclude-standard"):
            return "src/jacobian/math/graphs/_new.py\n"
        raise AssertionError(arguments)

    monkeypatch.setattr(runner, "_git", git)

    base, head, paths = runner.changed_paths(base="origin/main", repository=ROOT)

    assert base == "a" * 40
    assert head == "b" * 40
    assert paths == (
        "src/jacobian/math/graphs/_new.py",
        "src/jacobian/math/graphs/_operations.py",
        "src/jacobian/math/graphs/values.py",
        "tests/math/graphs/test_graph_coloring.py",
    )


def test_git_metadata_queries_use_the_bounded_operator_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load()
    observed: list[tuple[str, tuple[str, ...], Path, float]] = []

    def run_operator(
        command: str,
        arguments: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: float,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
        environment: dict[str, str],
    ) -> ToolCommandResult:
        assert environment["PATH"]
        assert stdout_limit_bytes == 4 * 1024 * 1024
        assert stderr_limit_bytes == 1024 * 1024
        observed.append((command, arguments, cwd, timeout_seconds))
        return ToolCommandResult(
            status=ToolCommandStatus.EXITED,
            exit_code=0,
            stdout=b"deadbeef\n",
            stderr=b"",
        )

    monkeypatch.setattr(runner, "run_operator_command", run_operator)

    assert runner._git("rev-parse", "HEAD", repository=ROOT) == "deadbeef"
    assert observed == [("git", ("rev-parse", "HEAD"), ROOT, 30.0)]


def test_selected_commands_use_the_bounded_operator_runner(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _load()
    observed: list[tuple[str, tuple[str, ...], Path, float]] = []

    def run_operator(
        command: str,
        arguments: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: float,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
        environment: dict[str, str],
    ) -> ToolCommandResult:
        assert environment["PATH"]
        assert stdout_limit_bytes == stderr_limit_bytes == 64 * 1024 * 1024
        observed.append((command, arguments, cwd, timeout_seconds))
        return ToolCommandResult(
            status=ToolCommandStatus.EXITED,
            exit_code=0,
            stdout=b"selected output\n",
            stderr=b"selected diagnostics\n",
        )

    monkeypatch.setattr(runner, "run_operator_command", run_operator)

    runner._run((("make", "test-math"),), repository=ROOT)

    assert observed == [("make", ("test-math",), ROOT, 30 * 60)]
    captured = capsys.readouterr()
    assert "selected output" in captured.out
    assert "selected diagnostics" in captured.err
