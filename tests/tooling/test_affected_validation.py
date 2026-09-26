"""Tests for local execution of the checked-in pull-request test plan."""

from __future__ import annotations

import importlib.util
import shlex
import sys
from collections.abc import Callable
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
        changed_paths=["src/jacobian/math/combinatorics/codes/general/_models.py"],
        repository=ROOT,
    )

    commands = runner.commands_for_plan(
        plan,
        paths=["src/jacobian/math/combinatorics/codes/general/_models.py"],
        repository=ROOT,
    )

    assert commands == (
        (
            "make",
            "lint-scoped",
            "PATHS=src/jacobian/math/combinatorics/codes/general/_models.py",
        ),
        (
            "make",
            "typecheck-scoped",
            "PATHS=src/jacobian/math/combinatorics/codes/general/_models.py",
        ),
        ("make", "test-math", "TESTS=tests/math/combinatorics/codes/general"),
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


def test_qepcad_owner_selects_qepcad_without_singular_replay() -> None:
    runner = _load()
    path = "src/jacobian/math/polynomials/real_algebra/_plane_components.py"
    plan = runner.build_plan(
        event="pull_request",
        base_revision="a" * 40,
        head_revision="b" * 40,
        changed_paths=[path],
        repository=ROOT,
    )

    commands = runner.commands_for_plan(plan, paths=[path], repository=ROOT)

    assert ("make", "test-qepcad") in commands
    assert ("make", "test-singular") not in commands


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
    monkeypatch.setenv("MATH_WORKERS", "2")
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
        stdout_sink: Callable[[bytes], None],
        stderr_sink: Callable[[bytes], None],
    ) -> ToolCommandResult:
        assert environment["PATH"]
        assert environment["MATH_WORKERS"] == "2"
        assert stdout_limit_bytes == stderr_limit_bytes == 64 * 1024 * 1024
        observed.append((command, arguments, cwd, timeout_seconds))
        stdout_sink(b"selected output\n")
        stderr_sink(b"selected diagnostics\n")
        live = capsys.readouterr()
        assert "[1/1] + make test-math" in live.out
        assert "selected output" in live.out
        assert "selected diagnostics" in live.err
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
    assert "selected output" not in captured.out
    assert "selected diagnostics" not in captured.err
    assert "[1/1] passed in" in captured.out
    assert "Validation complete: 1 commands passed" in captured.out


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [(ToolCommandStatus.EXITED, 2), (ToolCommandStatus.TIMED_OUT, None)],
)
def test_failed_command_stops_the_plan(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: ToolCommandStatus,
    exit_code: int | None,
) -> None:
    runner = _load()
    calls = []

    def run_operator(
        command: str, arguments: object, **kwargs: object
    ) -> ToolCommandResult:
        calls.append(command)
        return ToolCommandResult(
            status=status, exit_code=exit_code, stdout=b"", stderr=b""
        )

    monkeypatch.setattr(runner, "run_operator_command", run_operator)
    with pytest.raises(SystemExit, match="make test-math failed after"):
        runner._run((("make", "test-math"), ("make", "test-catalog")), repository=ROOT)
    assert calls == ["make"]
    assert "Validation complete" not in capsys.readouterr().out


def test_reproduction_preserves_quoted_selectors_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    runner = _load()
    monkeypatch.setenv("MATH_WORKERS", "1")
    monkeypatch.setenv("PYTEST_ARGS", "-k 'slow or exact'")
    monkeypatch.setenv("PRIVATE_TOKEN", "not-for-output")

    def fail(*args: object, **kwargs: object) -> ToolCommandResult:
        return ToolCommandResult(
            status=ToolCommandStatus.EXITED, exit_code=2, stdout=b"", stderr=b""
        )

    monkeypatch.setattr(runner, "run_operator_command", fail)
    repository = tmp_path / "checkout with spaces"
    command = ("make", "test-math", "TESTS=tests/math/a.py tests/math/b.py")
    with pytest.raises(SystemExit):
        runner._run((command,), repository=repository)
    diagnostic = capsys.readouterr().err
    reproduction = diagnostic.split("Reproduce: ", 1)[1].strip()
    tokens = shlex.split(reproduction)
    assert tokens[:4] == ["cd", str(repository), "&&", "env"]
    assert tokens[-3:] == list(command)
    assert "MATH_WORKERS=1" in tokens
    assert "PYTEST_ARGS=-k 'slow or exact'" in tokens
    assert "not-for-output" not in diagnostic
