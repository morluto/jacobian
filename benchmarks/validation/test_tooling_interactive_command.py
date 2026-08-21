from __future__ import annotations

import json
import os
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
from tools.command_runner import (
    ToolInteractiveCommand,
    ToolInteractiveRequest,
    ToolInteractiveStatus,
)

_PYTHON = sys.executable
_ENV = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC", "PATH": os.environ.get("PATH", "")}


def _echo_repl_script(tmp_path: Path) -> Path:
    """A minimal line-delimited request/response echo REPL for round-trip tests."""
    script = tmp_path / "echo_repl.py"
    script.write_text(
        textwrap.dedent(
            """\
            import json
            import sys
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                sys.stdout.write(json.dumps({"echo": payload}) + "\\n\\n")
                sys.stdout.flush()
            """
        ),
        encoding="utf-8",
    )
    return script


def test_interactive_request_rejects_bare_executable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        ToolInteractiveRequest(
            executable="python",
            cwd=str(tmp_path),
            startup_timeout_seconds=1.0,
            read_timeout_seconds=1.0,
        )


def test_interactive_request_rejects_relative_cwd(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute path"):
        ToolInteractiveRequest(
            executable=_PYTHON,
            cwd="relative",
            startup_timeout_seconds=1.0,
            read_timeout_seconds=1.0,
        )


def test_interactive_request_rejects_nonpositive_startup_timeout(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="startup_timeout_seconds"):
        ToolInteractiveRequest(
            executable=_PYTHON,
            cwd=str(tmp_path),
            startup_timeout_seconds=0,
            read_timeout_seconds=1.0,
        )


def test_interactive_request_rejects_nonfinite_read_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="read_timeout_seconds"):
        ToolInteractiveRequest(
            executable=_PYTHON,
            cwd=str(tmp_path),
            startup_timeout_seconds=1.0,
            read_timeout_seconds=float("nan"),
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable path fixture")
def test_interactive_round_trip_sends_and_receives_frames(tmp_path: Path) -> None:
    script = _echo_repl_script(tmp_path)
    request = ToolInteractiveRequest(
        executable=_PYTHON,
        arguments=(str(script),),
        environment=_ENV,
        cwd=str(tmp_path),
        startup_timeout_seconds=5.0,
        read_timeout_seconds=5.0,
        shutdown_timeout_seconds=5.0,
    )
    command = ToolInteractiveCommand(request)
    command.start()
    assert command.status is ToolInteractiveStatus.STARTED
    try:
        command.send(json.dumps({"hello": "world"}))
        response = command.read_response()
        assert json.loads(response) == {"echo": {"hello": "world"}}
    finally:
        return_code = command.close()
    assert command.status is ToolInteractiveStatus.CLOSED
    assert return_code == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable path fixture")
def test_interactive_read_timeout_raises_and_classifies(tmp_path: Path) -> None:
    """A child that never responds triggers a typed timeout, not a hang."""
    script = tmp_path / "silent.py"
    script.write_text("import sys; sys.stdin.read()\n", encoding="utf-8")
    request = ToolInteractiveRequest(
        executable=_PYTHON,
        arguments=(str(script),),
        environment=_ENV,
        cwd=str(tmp_path),
        startup_timeout_seconds=0.5,
        read_timeout_seconds=5.0,
        shutdown_timeout_seconds=5.0,
    )
    command = ToolInteractiveCommand(request)
    command.start()
    try:
        command.send(json.dumps({"ping": 1}))
        with pytest.raises(TimeoutError, match="timed out"):
            command.read_response()
        assert command.status is ToolInteractiveStatus.TIMED_OUT
    finally:
        command.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable path fixture")
def test_interactive_start_failure_classifies(tmp_path: Path) -> None:
    request = ToolInteractiveRequest(
        executable=str(tmp_path / "does-not-exist"),
        environment=_ENV,
        cwd=str(tmp_path),
        startup_timeout_seconds=1.0,
        read_timeout_seconds=1.0,
    )
    command = ToolInteractiveCommand(request)
    with pytest.raises(OSError):
        command.start()
    assert command.status is ToolInteractiveStatus.START_FAILED


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable path fixture")
def test_interactive_cancellation_event_aborts_read(tmp_path: Path) -> None:
    script = tmp_path / "silent.py"
    script.write_text("import sys; sys.stdin.read()\n", encoding="utf-8")
    cancel = threading.Event()
    request = ToolInteractiveRequest(
        executable=_PYTHON,
        arguments=(str(script),),
        environment=_ENV,
        cwd=str(tmp_path),
        startup_timeout_seconds=5.0,
        read_timeout_seconds=30.0,
        shutdown_timeout_seconds=5.0,
        cancellation_event=cancel,
    )
    command = ToolInteractiveCommand(request)
    command.start()
    try:
        command.send(json.dumps({"ping": 1}))
        cancel.set()
        with pytest.raises(TimeoutError, match="cancelled"):
            command.read_response()
        assert command.status is ToolInteractiveStatus.CANCELLED
    finally:
        command.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable path fixture")
def test_interactive_stderr_is_bounded(tmp_path: Path) -> None:
    script = tmp_path / "noisy.py"
    script.write_text(
        textwrap.dedent(
            """\
            import sys
            sys.stderr.write("x" * 10000)
            sys.stderr.flush()
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                sys.stdout.write(line + "\\n\\n")
                sys.stdout.flush()
            """
        ),
        encoding="utf-8",
    )
    request = ToolInteractiveRequest(
        executable=_PYTHON,
        arguments=(str(script),),
        environment=_ENV,
        cwd=str(tmp_path),
        startup_timeout_seconds=5.0,
        read_timeout_seconds=5.0,
        shutdown_timeout_seconds=5.0,
        stderr_limit_bytes=64,
    )
    command = ToolInteractiveCommand(request)
    command.start()
    try:
        # The child flushes oversized stderr before reading stdin; wait until
        # the drain thread observes it so read_response does not race ahead of
        # the bound flag under parallel host-validation load.
        deadline = time.monotonic() + 5.0
        while not command.stderr_exceeded and time.monotonic() < deadline:
            time.sleep(0.01)
        assert command.stderr_exceeded is True
        command.send("ping")
        with pytest.raises(RuntimeError, match="stderr bounds"):
            command.read_response()
    finally:
        command.close()
    assert len(command.stderr) <= 64
    assert command.stderr_exceeded is True
