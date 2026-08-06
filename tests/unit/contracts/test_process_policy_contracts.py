"""Pure validation contracts for the process-policy request models."""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from jacobian.bounded_process import ProcessPlatformTools
from jacobian.process_policy import ProcessRequest, resolve_process_platform_tools

_CWD = str(Path.cwd())
_ENV = dict(os.environ)


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
