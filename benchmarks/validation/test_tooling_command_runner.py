from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest
from tools.command_runner import (
    ToolCommandRequest,
    ToolCommandResult,
    ToolCommandStatus,
    ToolResolver,
    git_head_sha,
    operator_environment,
    run_operator_command,
    run_tool_command,
)


@pytest.mark.parametrize("stdout", [b"x" * 40, b"\xff" * 40])
def test_git_head_sha_rejects_non_digest_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: bytes
) -> None:
    monkeypatch.setattr(
        "tools.command_runner.run_operator_command",
        lambda *_args, **_kwargs: ToolCommandResult(
            status=ToolCommandStatus.EXITED,
            exit_code=0,
            stdout=stdout,
            stderr=b"",
        ),
    )

    assert git_head_sha(tmp_path) is None


def test_tooling_request_rejects_bare_executable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        ToolCommandRequest(
            executable="git",
            cwd=str(tmp_path),
            timeout_seconds=1.0,
            stdout_limit_bytes=1,
            stderr_limit_bytes=1,
        )


def test_operator_environment_does_not_forward_sensitive_host_values() -> None:
    environment = operator_environment(
        source={
            "LANG": "C",
            "PATH": "/poisoned",
            "PYTHONPATH": "/poisoned-python",
            "HTTPS_PROXY": "https://credential.invalid",
            "AWS_SECRET_ACCESS_KEY": "secret",
        }
    )
    assert environment == {"LANG": "C"}


def test_resolver_returns_absolute_executable_once() -> None:
    resolver = ToolResolver(search_path=str(Path(sys.executable).parent))
    resolved = resolver.resolve(Path(sys.executable).name)
    assert resolved is not None
    assert Path(resolved).is_absolute()
    assert resolver.resolve(Path(sys.executable).name) == resolved


def test_missing_operator_command_is_classified(tmp_path: Path) -> None:
    result = run_operator_command(
        "jacobian-command-that-does-not-exist",
        cwd=tmp_path,
        timeout_seconds=1.0,
    )
    assert result.status is ToolCommandStatus.START_FAILED
    assert result.exit_code is None
    assert result.diagnostic is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable path fixture")
def test_operator_command_has_bounded_output(tmp_path: Path) -> None:
    result = run_operator_command(
        Path(sys.executable).name,
        ("-c", "print('x' * 10000)"),
        cwd=tmp_path,
        timeout_seconds=5.0,
        stdout_limit_bytes=32,
        stderr_limit_bytes=32,
    )
    assert result.status is ToolCommandStatus.OUTPUT_LIMIT_EXCEEDED
    assert len(result.stdout) <= 32


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable path fixture")
def test_tool_command_streams_stdout_and_stderr_before_exit(tmp_path: Path) -> None:
    stdout = bytearray()
    stderr = bytearray()
    delivered = threading.Event()

    def receive_stdout(block: bytes) -> None:
        stdout.extend(block)
        delivered.set()

    request = ToolCommandRequest(
        executable=sys.executable,
        arguments=(
            "-c",
            "import sys,time; print('live', flush=True); "
            "print('warning', file=sys.stderr, flush=True); time.sleep(0.5)",
        ),
        cwd=str(tmp_path),
        timeout_seconds=5.0,
        stdout_limit_bytes=128,
        stderr_limit_bytes=128,
        stdout_sink=receive_stdout,
        stderr_sink=stderr.extend,
    )
    result_box: list[ToolCommandResult] = []
    runner = threading.Thread(
        target=lambda: result_box.append(run_tool_command(request))
    )

    runner.start()
    assert delivered.wait(timeout=2.0)
    assert runner.is_alive()
    runner.join(timeout=5.0)

    assert not runner.is_alive()
    assert result_box[0].status is ToolCommandStatus.EXITED
    assert stdout == b"live\n"
    assert stderr == b"warning\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable path fixture")
def test_streamed_output_remains_bounded(tmp_path: Path) -> None:
    streamed = bytearray()
    result = run_tool_command(
        ToolCommandRequest(
            executable=sys.executable,
            arguments=("-c", "print('x' * 10000, flush=True)"),
            cwd=str(tmp_path),
            timeout_seconds=5.0,
            stdout_limit_bytes=32,
            stderr_limit_bytes=32,
            stdout_sink=streamed.extend,
        )
    )

    assert result.status is ToolCommandStatus.OUTPUT_LIMIT_EXCEEDED
    assert streamed == result.stdout
    assert len(streamed) == 32


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable path fixture")
def test_tool_command_timeout_is_preserved_with_streaming(tmp_path: Path) -> None:
    streamed = bytearray()
    result = run_tool_command(
        ToolCommandRequest(
            executable=sys.executable,
            arguments=("-c", "import time; print('ready', flush=True); time.sleep(5)"),
            cwd=str(tmp_path),
            timeout_seconds=0.5,
            stdout_limit_bytes=128,
            stderr_limit_bytes=128,
            stdout_sink=streamed.extend,
        )
    )

    assert result.status is ToolCommandStatus.TIMED_OUT
    assert streamed == b"ready\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group fixture")
def test_reader_timeout_kills_descendants_that_hold_output_pipes(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "descendant.pid"
    descendant = "import time; time.sleep(30)"
    root = (
        "import pathlib, subprocess, sys; "
        f"child = subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))"
    )

    result = run_tool_command(
        ToolCommandRequest(
            executable=sys.executable,
            arguments=("-c", root),
            cwd=str(tmp_path),
            timeout_seconds=0.4,
            stdout_limit_bytes=128,
            stderr_limit_bytes=128,
        )
    )

    assert result.status is ToolCommandStatus.TIMED_OUT
    child_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        raise AssertionError(f"descendant (pid={child_pid}) survived reader timeout")


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable path fixture")
def test_sink_failure_stops_the_tool_process(tmp_path: Path) -> None:
    def reject_output(_block: bytes) -> None:
        raise RuntimeError("sink unavailable")

    result = run_tool_command(
        ToolCommandRequest(
            executable=sys.executable,
            arguments=("-c", "import time; print('ready', flush=True); time.sleep(5)"),
            cwd=str(tmp_path),
            timeout_seconds=4.0,
            stdout_limit_bytes=128,
            stderr_limit_bytes=128,
            stdout_sink=reject_output,
        )
    )

    assert result.status is ToolCommandStatus.STREAM_FAILED
    assert result.exit_code is None
    assert result.diagnostic == "stdout sink failed: RuntimeError"
