"""Lifecycle evidence for the repository command runner."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from tools import command_runner
from tools.command_runner import (
    ToolCommandRequest,
    ToolCommandStatus,
    ToolInteractiveCommand,
    ToolInteractiveRequest,
    ToolInteractiveStatus,
    run_tool_command,
)


def test_missing_process_pipe_is_rejected_and_child_is_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name == "posix",
    )

    class MissingStdout:
        pid = child.pid
        stdin = child.stdin
        stdout = None
        stderr = child.stderr
        returncode: int | None = None

        def poll(self) -> int | None:
            return child.poll()

        def kill(self) -> None:
            child.kill()

        def wait(self, timeout: float | None = None) -> int:
            return child.wait(timeout=timeout)

    monkeypatch.setattr(
        command_runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: MissingStdout(),
    )

    request = ToolCommandRequest(
        executable=str(Path(sys.executable).resolve()),
        arguments=("-c", "raise SystemExit(0)"),
        environment=dict(os.environ),
        cwd=str(tmp_path),
        timeout_seconds=1,
        stdout_limit_bytes=4096,
        stderr_limit_bytes=4096,
    )
    with pytest.raises(RuntimeError, match="stdout pipe was not created"):
        run_tool_command(request)

    assert child.poll() is not None


def test_command_timeout_includes_launch_execution_and_reaping(tmp_path: Path) -> None:
    request = ToolCommandRequest(
        executable=str(Path(sys.executable).resolve()),
        arguments=("-c", "import time; time.sleep(30)"),
        environment=dict(os.environ),
        cwd=str(tmp_path),
        timeout_seconds=0.4,
        stdout_limit_bytes=4096,
        stderr_limit_bytes=4096,
    )

    started = time.monotonic()
    result = run_tool_command(request)

    assert result.status is ToolCommandStatus.TIMED_OUT
    assert time.monotonic() - started < 0.8


def test_interactive_send_is_bounded_when_child_does_not_read(tmp_path: Path) -> None:
    command = ToolInteractiveCommand(
        ToolInteractiveRequest(
            executable=str(Path(sys.executable).resolve()),
            arguments=("-c", "import time; time.sleep(30)"),
            environment=dict(os.environ),
            cwd=str(tmp_path),
            startup_timeout_seconds=0.2,
            read_timeout_seconds=0.2,
            shutdown_timeout_seconds=0.2,
            stdout_limit_bytes=4096,
            stderr_limit_bytes=4096,
        )
    )
    command.start()

    started = time.monotonic()
    try:
        command.send("x" * (2 * 1024 * 1024))
    except TimeoutError:
        pass
    else:  # pragma: no cover - a pipe unexpectedly accepting all input is unsafe
        raise AssertionError(
            "interactive write should time out when the child does not read"
        )

    assert command.status is ToolInteractiveStatus.TIMED_OUT
    assert time.monotonic() - started < 0.6
