"""Immutable process-policy contracts and thin gateway for bounded execution.

This module contains the policy layer and re-exports selected engine types:

* :class:`ProcessTermination` — the normalized outcome enum.
* :class:`ProcessPlatformTools` — bootstrap-resolved absolute helper paths.
* :class:`ProcessRequest` — the immutable, validated execution request
  (absolute executable, arguments excluding the executable, explicit absolute
  cwd, explicit environment, positive finite timeout, non-negative integer
  output limits, optional resource limits, and an explicit cancellation
  signal).
* :class:`ProcessResult` — the normalized outcome.
* :func:`execute_process` — a thin gateway that delegates to
  :func:`jacobian.bounded_process.run_bounded_process` (the sole low-level
  engine) and normalizes its result and start-up :class:`OSError` into
  :attr:`ProcessTermination.START_FAILED`.
* :class:`BoundedInteractiveProcess`, :class:`InteractiveProcessError`,
  :class:`InteractiveProcessRequest` — re-exported interactive-process engine
  types used by the Lean REPL transport.

No Popen, capture, kill, or ``resource`` code lives here.  There is no second
:class:`ProcessResourceLimits`; the engine's definition is re-exported.
Shell strings and ``shell=True`` are structurally impossible because the
gateway always passes an explicit ``[executable, *arguments]`` list.
"""

from __future__ import annotations

import math
import os
import shutil
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from jacobian.bounded_process import (
    BoundedInteractiveProcess,
    BoundedProcessResult,
    InteractiveProcessError,
    InteractiveProcessRequest,
    ProcessPlatformTools,
    ProcessResourceLimits,
    run_bounded_process,
)

__all__ = [
    "BoundedInteractiveProcess",
    "InteractiveProcessError",
    "InteractiveProcessRequest",
    "ProcessPlatformTools",
    "ProcessRequest",
    "ProcessResourceLimits",
    "ProcessResult",
    "ProcessTermination",
    "execute_process",
    "resolve_process_platform_tools",
]


def resolve_process_platform_tools() -> ProcessPlatformTools:
    """Resolve operating-system helpers once during runtime bootstrap."""

    prlimit: str | None = None
    taskkill: str | None = None
    if os.name == "posix":
        candidate = shutil.which("prlimit")
        if candidate is not None:
            prlimit = str(Path(candidate).resolve(strict=True))
    elif os.name == "nt":  # pragma: no cover - exercised in Windows CI
        candidate = shutil.which("taskkill")
        if candidate is not None:
            taskkill = str(Path(candidate).resolve(strict=True))
    return ProcessPlatformTools(
        prlimit_executable=prlimit,
        taskkill_executable=taskkill,
    )


class ProcessTermination(StrEnum):
    """Normalized outcome of a bounded process execution."""

    EXITED = "EXITED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"
    OUTPUT_LIMIT_EXCEEDED = "OUTPUT_LIMIT_EXCEEDED"
    START_FAILED = "START_FAILED"


@dataclass(frozen=True, slots=True)
class ProcessRequest:
    """Immutable, validated description of one bounded subprocess execution.

    The gateway spawns ``[executable, *arguments]`` in *cwd* with
    *environment*, feeds *stdin_bytes* on stdin, and enforces
    *timeout_seconds* plus per-stream output limits.  Optional
    *resource_limits* activate prlimit wrapping only when at least one field
    is active.  *cancellation_event* is the cooperative cancellation signal
    consulted by the engine; it is part of the request so callers cannot
    forget to wire cancellation.
    """

    executable: str
    arguments: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    cwd: str = ""
    timeout_seconds: float = 0.0
    stdin_bytes: bytes = b""
    stdout_limit_bytes: int = 0
    stderr_limit_bytes: int = 0
    resource_limits: ProcessResourceLimits | None = None
    cancellation_event: threading.Event | None = None

    def __post_init__(self) -> None:
        self._validate_command()
        self._validate_io_policy()

    def _validate_command(self) -> None:
        if not self.executable:
            raise ValueError("executable must be non-empty")
        if not Path(self.executable).is_absolute():
            raise ValueError("executable must be an absolute path")
        if not isinstance(self.arguments, tuple) or not all(
            isinstance(arg, str) for arg in self.arguments
        ):
            raise ValueError("arguments must be a tuple of strings")
        if not isinstance(self.environment, Mapping):
            raise ValueError("environment must be a mapping")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.environment.items()
        ):
            raise ValueError("environment keys and values must be strings")
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )
        if not self.cwd:
            raise ValueError("cwd must be an explicit absolute path")
        if not Path(self.cwd).is_absolute():
            raise ValueError("cwd must be an absolute path")

    def _validate_io_policy(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout must be positive and finite")
        if not isinstance(self.stdin_bytes, bytes):
            raise ValueError("stdin_bytes must be bytes")
        for name in ("stdout_limit_bytes", "stderr_limit_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.cancellation_event is not None and not isinstance(
            self.cancellation_event, threading.Event
        ):
            raise ValueError("cancellation_event must be a threading.Event or None")


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Normalized outcome of :func:`execute_process`.

    *termination* is exactly one :class:`ProcessTermination`.  *returncode* is
    the exit status when the process exited, otherwise ``None``.  Captured
    output is truncated to the respective limit; the ``*_exceeded`` flags
    report whether the limit was hit.
    """

    termination: ProcessTermination
    returncode: int | None
    stdout: bytes
    stderr: bytes
    stdout_exceeded: bool
    stderr_exceeded: bool
    peak_rss_bytes: int | None = None


def _normalize_termination(completed: BoundedProcessResult) -> ProcessTermination:
    if completed.cancelled:
        return ProcessTermination.CANCELLED
    if completed.timed_out:
        return ProcessTermination.TIMED_OUT
    if completed.stdout_exceeded or completed.stderr_exceeded:
        return ProcessTermination.OUTPUT_LIMIT_EXCEEDED
    return ProcessTermination.EXITED


_PLATFORM_TOOLS = resolve_process_platform_tools()


def execute_process(request: ProcessRequest) -> ProcessResult:
    """Run a bounded child using the platform helpers resolved at bootstrap.

    This is a thin gateway: it builds the ``[executable, *arguments]`` command
    list, delegates to :func:`jacobian.bounded_process.run_bounded_process`
    (the sole low-level engine), and normalizes the result.  A start-up
    :class:`OSError` from the engine (missing executable, permission denied)
    is normalized to :attr:`ProcessTermination.START_FAILED` instead of
    propagating.
    """

    command: Sequence[str] = [request.executable, *request.arguments]
    try:
        completed = run_bounded_process(
            command,
            input_bytes=request.stdin_bytes,
            timeout_seconds=request.timeout_seconds,
            environment=request.environment,
            stdout_limit=request.stdout_limit_bytes,
            stderr_limit=request.stderr_limit_bytes,
            resource_limits=request.resource_limits,
            cwd=request.cwd,
            platform_tools=_PLATFORM_TOOLS,
            cancellation_event=request.cancellation_event,
        )
    except OSError:
        return ProcessResult(
            termination=ProcessTermination.START_FAILED,
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            peak_rss_bytes=None,
        )
    return ProcessResult(
        termination=_normalize_termination(completed),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        stdout_exceeded=completed.stdout_exceeded,
        stderr_exceeded=completed.stderr_exceeded,
        peak_rss_bytes=completed.peak_rss_bytes,
    )
