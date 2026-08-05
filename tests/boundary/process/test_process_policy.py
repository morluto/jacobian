"""Tests for the PR4 process-policy foundation.

These tests prove:

* :mod:`jacobian.process_policy` contains no duplicate engine (no Popen,
  capture, or kill code) and delegates to :func:`run_bounded_process`.
* The module imports cleanly on Windows (no unconditional ``resource`` or
  platform-only import at module load).
* :class:`ProcessRequest` enforces absolute executable, explicit absolute cwd,
  explicit environment, positive finite timeout, non-negative integer output
  limits, and carries the cancellation signal.
* :class:`ProcessPlatformTools` requires absolute paths and rejects relative
  ones.
* :func:`execute_process` normalizes a start-up :class:`OSError` to
  :attr:`ProcessTermination.START_FAILED`.
* The engine never wraps with prlimit when no resource limit is active.
* End-to-end execution through the gateway produces a normalized
  :class:`ProcessResult`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from jacobian.bounded_process import (
    BoundedInteractiveProcess,
    BoundedProcessResult,
    InteractiveProcessError,
    InteractiveProcessRequest,
    ProcessPlatformTools,
    ProcessResourceLimits,
    run_bounded_process,
)
from jacobian.process_policy import (
    ProcessRequest,
    ProcessTermination,
    execute_process,
    resolve_process_platform_tools,
)

_CWD = str(Path.cwd())
_ENV = dict(os.environ)


# ---------------------------------------------------------------------------
# Delegation to the bounded engine.
# ---------------------------------------------------------------------------


def test_process_policy_has_no_second_process_resource_limits() -> None:
    """ProcessResourceLimits must be defined once (in bounded_process) and
    re-exported, not duplicated."""

    from jacobian.bounded_process import ProcessResourceLimits as EngineLimits
    from jacobian.process_policy import ProcessResourceLimits as PolicyLimits

    assert PolicyLimits is EngineLimits


def test_execute_process_delegates_to_run_bounded_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """execute_process must call run_bounded_process and not spawn directly."""

    sentinel = BoundedProcessResult(
        returncode=0,
        stdout=b"ok",
        stderr=b"",
        stdout_exceeded=False,
        stderr_exceeded=False,
        timed_out=False,
        cancelled=False,
    )
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        captured["command"] = command
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr("jacobian.process_policy.run_bounded_process", fake_run)
    request = ProcessRequest(
        executable=sys.executable,
        arguments=("-c", "pass"),
        environment=_ENV,
        cwd=_CWD,
        timeout_seconds=1.0,
        stdout_limit_bytes=4096,
        stderr_limit_bytes=4096,
    )
    result = execute_process(request)
    assert result.termination is ProcessTermination.EXITED
    assert result.returncode == 0
    assert result.stdout == b"ok"
    assert captured["command"] == [sys.executable, "-c", "pass"]
    assert captured["kwargs"]["cwd"] == _CWD


# ---------------------------------------------------------------------------
# Windows-safe import: no unconditional resource/platform-only import.
# ---------------------------------------------------------------------------


def test_process_policy_imports_without_resource_module() -> None:
    """Importing process_policy must not require the POSIX ``resource`` module.

    This proves Windows-safe import: the module loads even when ``resource``
    is absent.  We simulate the absence by blocking it in a fresh interpreter.
    """

    script = textwrap.dedent(
        """
        import sys
        # Block the POSIX-only resource module as if on Windows.
        import importlib.abc
        class _BlockResource(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "resource":
                    raise ImportError("blocked for test")
                return None
        sys.meta_path.insert(0, _BlockResource())
        # Remove any cached import.
        sys.modules.pop("resource", None)
        import jacobian.process_policy
        print("imported-ok")
        """,
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "imported-ok"


def test_bounded_process_imports_without_resource_module() -> None:
    """bounded_process must also import cleanly when ``resource`` is blocked.

    The engine imports ``resource`` lazily inside ``_apply_resource_limits``,
    so module load must not require it.
    """

    script = textwrap.dedent(
        """
        import sys
        import importlib.abc
        class _BlockResource(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "resource":
                    raise ImportError("blocked for test")
                return None
        sys.meta_path.insert(0, _BlockResource())
        sys.modules.pop("resource", None)
        import jacobian.bounded_process
        print("imported-ok")
        """,
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "imported-ok"


# ---------------------------------------------------------------------------
# ProcessRequest validation.
# ---------------------------------------------------------------------------


def _valid_request_kwargs() -> dict[str, object]:
    return {
        "executable": sys.executable,
        "arguments": ("-c", "pass"),
        "environment": _ENV,
        "cwd": _CWD,
        "timeout_seconds": 1.0,
        "stdout_limit_bytes": 4096,
        "stderr_limit_bytes": 4096,
    }


def test_request_rejects_relative_executable() -> None:
    kwargs = _valid_request_kwargs()
    kwargs["executable"] = "python3"
    with pytest.raises(ValueError, match="executable must be an absolute path"):
        ProcessRequest(**kwargs)


def test_request_rejects_empty_executable() -> None:
    kwargs = _valid_request_kwargs()
    kwargs["executable"] = ""
    with pytest.raises(ValueError, match="executable must be non-empty"):
        ProcessRequest(**kwargs)


def test_request_rejects_implicit_cwd() -> None:
    kwargs = _valid_request_kwargs()
    kwargs["cwd"] = ""
    with pytest.raises(ValueError, match="cwd must be an explicit absolute path"):
        ProcessRequest(**kwargs)


def test_request_rejects_relative_cwd() -> None:
    kwargs = _valid_request_kwargs()
    kwargs["cwd"] = "relative/path"
    with pytest.raises(ValueError, match="cwd must be an absolute path"):
        ProcessRequest(**kwargs)


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_request_rejects_invalid_timeout(timeout: float) -> None:
    kwargs = _valid_request_kwargs()
    kwargs["timeout_seconds"] = timeout
    with pytest.raises(ValueError, match="timeout must be positive"):
        ProcessRequest(**kwargs)


@pytest.mark.parametrize("limit", [-1, -100])
def test_request_rejects_negative_output_limits(limit: int) -> None:
    kwargs = _valid_request_kwargs()
    kwargs["stdout_limit_bytes"] = limit
    with pytest.raises(ValueError, match="stdout_limit_bytes"):
        ProcessRequest(**kwargs)


def test_request_rejects_non_integer_output_limit() -> None:
    kwargs = _valid_request_kwargs()
    kwargs["stdout_limit_bytes"] = 1.5  # type: ignore[assignment]
    with pytest.raises(ValueError, match="stdout_limit_bytes"):
        ProcessRequest(**kwargs)


def test_request_accepts_zero_output_limits() -> None:
    kwargs = _valid_request_kwargs()
    kwargs["stdout_limit_bytes"] = 0
    kwargs["stderr_limit_bytes"] = 0
    request = ProcessRequest(**kwargs)
    assert request.stdout_limit_bytes == 0
    assert request.stderr_limit_bytes == 0


def test_request_arguments_exclude_executable() -> None:
    request = ProcessRequest(
        executable=sys.executable,
        arguments=("-c", "print(1)"),
        environment=_ENV,
        cwd=_CWD,
        timeout_seconds=1.0,
        stdout_limit_bytes=4096,
        stderr_limit_bytes=4096,
    )
    assert request.executable == sys.executable
    assert request.arguments == ("-c", "print(1)")
    assert sys.executable not in request.arguments


def test_request_rejects_mutable_argument_sequence() -> None:
    kwargs = _valid_request_kwargs()
    kwargs["arguments"] = ["-c", "pass"]
    with pytest.raises(ValueError, match="tuple of strings"):
        ProcessRequest(**kwargs)


def test_request_rejects_non_string_environment_value() -> None:
    kwargs = _valid_request_kwargs()
    kwargs["environment"] = {"PORT": 8000}
    with pytest.raises(ValueError, match="keys and values must be strings"):
        ProcessRequest(**kwargs)


def test_request_rejects_boolean_output_limit() -> None:
    kwargs = _valid_request_kwargs()
    kwargs["stdout_limit_bytes"] = True
    with pytest.raises(ValueError, match="stdout_limit_bytes"):
        ProcessRequest(**kwargs)


def test_request_is_frozen() -> None:
    request = ProcessRequest(**_valid_request_kwargs())
    with pytest.raises(FrozenInstanceError):
        request.executable = "/bin/other"  # type: ignore[misc]


def test_request_carries_cancellation_signal() -> None:
    event = threading.Event()
    kwargs = _valid_request_kwargs()
    kwargs["cancellation_event"] = event
    request = ProcessRequest(**kwargs)
    assert request.cancellation_event is event


def test_request_rejects_non_event_cancellation() -> None:
    kwargs = _valid_request_kwargs()
    kwargs["cancellation_event"] = "not-an-event"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="cancellation_event"):
        ProcessRequest(**kwargs)


def test_request_environment_is_immutable_view() -> None:
    request = ProcessRequest(**_valid_request_kwargs())
    with pytest.raises(TypeError):
        request.environment["__test"] = "value"  # type: ignore[index]


# ---------------------------------------------------------------------------
# ProcessPlatformTools absoluteness.
# ---------------------------------------------------------------------------


def test_platform_tools_rejects_relative_prlimit() -> None:
    with pytest.raises(ValueError, match="prlimit_executable must be an absolute path"):
        ProcessPlatformTools(prlimit_executable="prlimit")


def test_platform_tools_rejects_relative_taskkill() -> None:
    with pytest.raises(
        ValueError, match="taskkill_executable must be an absolute path"
    ):
        ProcessPlatformTools(taskkill_executable="taskkill")


def test_platform_tools_resolver_returns_absolute_or_none() -> None:
    tools = resolve_process_platform_tools()
    if tools.prlimit_executable is not None:
        assert Path(tools.prlimit_executable).is_absolute()
    if tools.taskkill_executable is not None:
        assert Path(tools.taskkill_executable).is_absolute()


# ---------------------------------------------------------------------------
# Start-failure normalization.
# ---------------------------------------------------------------------------


def test_execute_process_normalizes_missing_executable_to_start_failed() -> None:
    request = ProcessRequest(
        executable="/nonexistent/absolute/path/does-not-exist",
        arguments=(),
        environment=_ENV,
        cwd=_CWD,
        timeout_seconds=1.0,
        stdout_limit_bytes=4096,
        stderr_limit_bytes=4096,
    )
    result = execute_process(request)
    assert result.termination is ProcessTermination.START_FAILED
    assert result.returncode is None
    assert result.stdout == b""
    assert result.stderr == b""


def test_execute_process_normalizes_non_executable_file_to_start_failed(
    tmp_path: Path,
) -> None:
    target = tmp_path / "not-a-program"
    target.write_text("not a program")
    request = ProcessRequest(
        executable=str(target),
        arguments=(),
        environment=_ENV,
        cwd=_CWD,
        timeout_seconds=1.0,
        stdout_limit_bytes=4096,
        stderr_limit_bytes=4096,
    )
    result = execute_process(request)
    assert result.termination is ProcessTermination.START_FAILED


# ---------------------------------------------------------------------------
# prlimit only when an active limit is set.
# ---------------------------------------------------------------------------


def test_resource_limited_command_does_not_wrap_when_no_active_limit() -> None:
    from jacobian.bounded_process import _resource_limited_command

    limits = ProcessResourceLimits()  # all None
    wrapped, applied = _resource_limited_command(
        [sys.executable, "-c", "pass"], limits, "/usr/bin/prlimit"
    )
    assert not applied
    assert wrapped == [sys.executable, "-c", "pass"]


def test_resource_limited_command_wraps_when_active_limit_present() -> None:
    from jacobian.bounded_process import _resource_limited_command

    limits = ProcessResourceLimits(cpu_seconds=2)
    wrapped, applied = _resource_limited_command(
        [sys.executable, "-c", "pass"], limits, "/usr/bin/prlimit"
    )
    assert applied
    assert wrapped[0] == "/usr/bin/prlimit"
    assert "--cpu=2:2" in wrapped
    assert wrapped[-3:] == ["--", sys.executable, "-c", "pass"][-3:]


# ---------------------------------------------------------------------------
# End-to-end gateway behavior.
# ---------------------------------------------------------------------------


def test_execute_process_returns_exited_for_successful_child() -> None:
    request = ProcessRequest(
        executable=sys.executable,
        arguments=("-c", "print('hello')"),
        environment=_ENV,
        cwd=_CWD,
        timeout_seconds=5.0,
        stdout_limit_bytes=4096,
        stderr_limit_bytes=4096,
    )
    result = execute_process(request)
    assert result.termination is ProcessTermination.EXITED
    assert result.returncode == 0
    assert result.stdout == b"hello\n"


def test_execute_process_returns_timed_out_for_sleeping_child() -> None:
    request = ProcessRequest(
        executable=sys.executable,
        arguments=("-c", "import time; time.sleep(30)"),
        environment=_ENV,
        cwd=_CWD,
        timeout_seconds=0.5,
        stdout_limit_bytes=4096,
        stderr_limit_bytes=4096,
    )
    started = time.monotonic()
    result = execute_process(request)
    assert result.termination is ProcessTermination.TIMED_OUT
    assert time.monotonic() - started < 5


def test_execute_process_returns_cancelled_when_event_set() -> None:
    event = threading.Event()
    timer = threading.Timer(0.2, event.set)
    timer.start()
    try:
        request = ProcessRequest(
            executable=sys.executable,
            arguments=("-c", "import time; time.sleep(30)"),
            environment=_ENV,
            cwd=_CWD,
            timeout_seconds=20.0,
            stdout_limit_bytes=4096,
            stderr_limit_bytes=4096,
            cancellation_event=event,
        )
        started = time.monotonic()
        result = execute_process(request)
    finally:
        timer.cancel()
    assert result.termination is ProcessTermination.CANCELLED
    assert time.monotonic() - started < 5


def test_execute_process_returns_output_limit_exceeded() -> None:
    request = ProcessRequest(
        executable=sys.executable,
        arguments=("-c", "print('x' * 10000)"),
        environment=_ENV,
        cwd=_CWD,
        timeout_seconds=5.0,
        stdout_limit_bytes=64,
        stderr_limit_bytes=4096,
    )
    result = execute_process(request)
    assert result.termination is ProcessTermination.OUTPUT_LIMIT_EXCEEDED
    assert result.stdout_exceeded
    assert len(result.stdout) <= 64


def test_interactive_response_cannot_hide_stderr_overflow(tmp_path: Path) -> None:
    request = InteractiveProcessRequest(
        executable=sys.executable,
        arguments=(
            "-c",
            (
                "import sys, time; sys.stdin.readline(); sys.stdin.readline(); "
                "sys.stderr.write('x' * 1024); sys.stderr.flush(); "
                "print('{}\\n', flush=True)"
            ),
        ),
        environment=_ENV,
        cwd=str(tmp_path),
        startup_timeout_seconds=2.0,
        read_timeout_seconds=2.0,
        stderr_limit_bytes=8,
    )
    process = BoundedInteractiveProcess(request)
    process.start()
    try:
        with pytest.raises(InteractiveProcessError, match="stderr limit"):
            process.exchange({"cmd": "check"})
    finally:
        process.close()


def test_execute_process_passes_cwd_to_child(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    request = ProcessRequest(
        executable=sys.executable,
        arguments=("-c", f"open(r'{marker}', 'w').write('ok')"),
        environment=_ENV,
        cwd=str(tmp_path),
        timeout_seconds=5.0,
        stdout_limit_bytes=4096,
        stderr_limit_bytes=4096,
    )
    result = execute_process(request)
    assert result.termination is ProcessTermination.EXITED
    assert marker.read_text() == "ok"


# ---------------------------------------------------------------------------
# Existing run_bounded_process behavior preserved.
# ---------------------------------------------------------------------------


def test_run_bounded_process_returns_bounded_result() -> None:
    completed = run_bounded_process(
        [sys.executable, "-c", "print('bounded')"],
        input_bytes=b"",
        timeout_seconds=5.0,
        environment=_ENV,
        stdout_limit=4096,
        stderr_limit=4096,
    )
    assert isinstance(completed, BoundedProcessResult)
    assert completed.returncode == 0
    assert completed.stdout == b"bounded\n"
    assert not completed.timed_out
    assert not completed.cancelled


def test_run_bounded_process_accepts_new_cwd_parameter(tmp_path: Path) -> None:
    marker = tmp_path / "cwd-marker"
    completed = run_bounded_process(
        [sys.executable, "-c", f"open(r'{marker}', 'w').write('ok')"],
        input_bytes=b"",
        timeout_seconds=5.0,
        environment=_ENV,
        stdout_limit=4096,
        stderr_limit=4096,
        cwd=str(tmp_path),
    )
    assert completed.returncode == 0
    assert marker.read_text() == "ok"


def test_run_bounded_process_accepts_platform_tools_parameter() -> None:
    tools = resolve_process_platform_tools()
    completed = run_bounded_process(
        [sys.executable, "-c", "print('tools')"],
        input_bytes=b"",
        timeout_seconds=5.0,
        environment=_ENV,
        stdout_limit=4096,
        stderr_limit=4096,
        platform_tools=tools,
    )
    assert completed.returncode == 0
    assert completed.stdout == b"tools\n"
