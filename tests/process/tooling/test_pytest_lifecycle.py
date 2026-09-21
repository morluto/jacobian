from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from tools import pytest_lifecycle
from tools.command_runner import (
    ToolCommandRequest,
    ToolCommandResult,
    ToolCommandStatus,
)


def _command_result(
    exit_code: int | None, *, status: ToolCommandStatus = ToolCommandStatus.EXITED
) -> ToolCommandResult:
    return ToolCommandResult(
        status=status,
        exit_code=exit_code,
        stdout=b"",
        stderr=b"",
    )


def test_real_pytest_child_succeeds_and_cleans_its_temp_tree(tmp_path: Path) -> None:
    test_file = tmp_path / "test_probe.py"
    test_file.write_text(
        "def test_probe():\n    assert 2 + 2 == 4\n",
        encoding="utf-8",
    )

    result = pytest_lifecycle.run_pytest(
        [str(test_file), "-q"],
        root=tmp_path,
        name="real-child",
        environment=dict(os.environ),
        timeout_seconds=30,
    )

    assert result.status is ToolCommandStatus.EXITED
    assert result.exit_code == 0
    assert result.retained is False
    assert not result.basetemp.parent.exists()


def test_success_uses_unique_worktree_basetemp_and_cleans_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[tuple[tuple[str, ...], Path]] = []

    def run(request: ToolCommandRequest) -> ToolCommandResult:
        assert request.executable == sys.executable
        observed.append((request.arguments, Path(request.cwd)))
        basetemp_argument = next(
            argument
            for argument in request.arguments
            if argument.startswith("--basetemp=")
        )
        basetemp = Path(basetemp_argument.split("=", 1)[1])
        (basetemp.parent / "session-template").mkdir()
        return _command_result(0)

    monkeypatch.setattr(pytest_lifecycle, "run_tool_command", run)

    first = pytest_lifecycle.run_pytest(
        ["tests/math/test_one.py"],
        root=tmp_path,
        name="unit",
        environment={},
    )
    second = pytest_lifecycle.run_pytest(
        ["tests/math/test_one.py"],
        root=tmp_path,
        name="unit",
        environment={},
    )

    assert first.basetemp != second.basetemp
    assert first.basetemp.is_relative_to(tmp_path / ".pytest_cache" / "basetemp")
    assert first.basetemp.name == "pytest"
    assert not first.basetemp.exists()
    assert not first.basetemp.parent.exists()
    assert not second.basetemp.exists()
    assert any(str(argument).startswith("--basetemp=") for argument in observed[0][0])


def test_failure_cleanup_is_default_and_retention_is_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pytest_lifecycle, "run_tool_command", lambda _request: _command_result(1)
    )

    cleaned = pytest_lifecycle.run_pytest(
        ["test_bad.py"], root=tmp_path, name="clean", environment={}
    )
    retained = pytest_lifecycle.run_pytest(
        ["test_bad.py"],
        root=tmp_path,
        name="retain",
        environment={},
        retain_on_failure=True,
    )

    assert not cleaned.basetemp.exists()
    assert retained.retained is True
    assert retained.basetemp.is_dir()


def test_timeout_reports_timed_out_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pytest_lifecycle,
        "run_tool_command",
        lambda _request: _command_result(None, status=ToolCommandStatus.TIMED_OUT),
    )

    result = pytest_lifecycle.run_pytest(
        ["test_slow.py"], root=tmp_path, name="slow", environment={}
    )

    assert result.status is ToolCommandStatus.TIMED_OUT
    assert result.exit_code == 1


def test_explicit_basetemp_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="owned"):
        pytest_lifecycle.run_pytest(
            ["--basetemp=/tmp/shared"],
            root=tmp_path,
            name="unsafe",
            environment={},
        )
