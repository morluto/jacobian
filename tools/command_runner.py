"""Bounded execution policy for repository and benchmark commands.

This module is deliberately independent of the product runtime. Repository
tooling resolves executables once, builds an allowlisted environment, and owns
its bounded child-process lifecycle here.
"""

from __future__ import annotations

import math
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

_OPERATOR_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
        "WINDIR",
    }
)
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class ToolCommandStatus(StrEnum):
    """Terminal state of one tooling command."""

    EXITED = "EXITED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"
    OUTPUT_LIMIT_EXCEEDED = "OUTPUT_LIMIT_EXCEEDED"
    STREAM_FAILED = "STREAM_FAILED"
    START_FAILED = "START_FAILED"


@dataclass(frozen=True, slots=True)
class ToolCommandRequest:
    """Immutable request for one bounded repository command."""

    executable: str
    arguments: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    cwd: str = ""
    timeout_seconds: float = 0.0
    stdin_bytes: bytes = b""
    stdout_limit_bytes: int = 0
    stderr_limit_bytes: int = 0
    cancellation_event: threading.Event | None = None
    stdout_sink: Callable[[bytes], None] | None = field(
        default=None, compare=False, repr=False
    )
    stderr_sink: Callable[[bytes], None] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        self._validate_command()
        self._validate_io_policy()

    def _validate_command(self) -> None:
        if not Path(self.executable).is_absolute():
            raise ValueError("tool executable must be an absolute path")
        object.__setattr__(self, "arguments", tuple(self.arguments))
        if not all(isinstance(argument, str) for argument in self.arguments):
            raise TypeError("tool arguments must be strings")
        if not isinstance(self.environment, Mapping):
            raise TypeError("tool environment must be a mapping")
        environment = dict(self.environment)
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        ):
            raise TypeError("tool environment keys and values must be strings")
        object.__setattr__(self, "environment", MappingProxyType(environment))
        if not self.cwd or not Path(self.cwd).is_absolute():
            raise ValueError("tool working directory must be an explicit absolute path")

    def _validate_io_policy(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("tool timeout must be positive and finite")
        for name in ("stdout_limit_bytes", "stderr_limit_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.cancellation_event is not None and not isinstance(
            self.cancellation_event, threading.Event
        ):
            raise TypeError("cancellation_event must be a threading.Event or None")
        for name in ("stdout_sink", "stderr_sink"):
            sink = getattr(self, name)
            if sink is not None and not callable(sink):
                raise TypeError(f"{name} must be callable or None")


@dataclass(frozen=True, slots=True)
class ToolCommandResult:
    """Bounded output and normalized terminal state for a tooling command."""

    status: ToolCommandStatus
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    stdout_exceeded: bool = False
    stderr_exceeded: bool = False
    diagnostic: str | None = None


class ToolResolver:
    """Resolve operator-installed commands to stable absolute paths."""

    def __init__(self, *, search_path: str | None = None) -> None:
        self._search_path = search_path
        self._resolved: dict[str, str] = {}

    def resolve(self, command: str) -> str | None:
        if not command or Path(command).name != command:
            raise ValueError("tool resolver accepts a bare command name")
        if command not in self._resolved:
            candidate = shutil.which(command, path=self._search_path)
            if candidate is None:
                return None
            self._resolved[command] = str(Path(candidate).resolve(strict=True))
        return self._resolved[command]


_DEFAULT_RESOLVER = ToolResolver()
_TASKKILL = _DEFAULT_RESOLVER.resolve("taskkill") if os.name == "nt" else None


def run_operator_command(
    command: str,
    arguments: Iterable[str] = (),
    *,
    cwd: Path,
    timeout_seconds: float = 60.0,
    stdout_limit_bytes: int = 4 * 1024 * 1024,
    stderr_limit_bytes: int = 4 * 1024 * 1024,
    environment: Mapping[str, str] | None = None,
) -> ToolCommandResult:
    """Resolve and execute one operator-installed command under tooling policy."""

    executable = _DEFAULT_RESOLVER.resolve(command)
    if executable is None:
        return ToolCommandResult(
            status=ToolCommandStatus.START_FAILED,
            exit_code=None,
            stdout=b"",
            stderr=b"",
            diagnostic=f"{command} is unavailable",
        )
    request = ToolCommandRequest(
        executable=executable,
        arguments=tuple(arguments),
        environment=(operator_environment() if environment is None else environment),
        cwd=str(cwd.resolve(strict=True)),
        timeout_seconds=timeout_seconds,
        stdout_limit_bytes=stdout_limit_bytes,
        stderr_limit_bytes=stderr_limit_bytes,
    )
    return run_tool_command(request)


def git_head_sha(root: Path) -> str | None:
    """Return the current Git commit through the bounded tooling boundary."""

    result = run_operator_command(
        "git", ("rev-parse", "HEAD"), cwd=root, timeout_seconds=30.0
    )
    if result.status is not ToolCommandStatus.EXITED or result.exit_code != 0:
        return None
    try:
        value = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError:
        return None
    return value if _GIT_SHA.fullmatch(value) is not None else None


def git_tracked_worktree_is_clean(root: Path) -> bool:
    """Return whether committed and staged source matches HEAD, ignoring untracked data."""

    result = run_operator_command(
        "git",
        ("status", "--porcelain=v1", "--untracked-files=no"),
        cwd=root,
        timeout_seconds=30.0,
        stdout_limit_bytes=1024 * 1024,
        stderr_limit_bytes=1024 * 1024,
    )
    return bool(
        result.status is ToolCommandStatus.EXITED
        and result.exit_code == 0
        and not result.stdout.strip()
    )


def operator_environment(
    *,
    source: Mapping[str, str] | None = None,
    declared: Mapping[str, str] | None = None,
    include: Iterable[str] = (),
) -> Mapping[str, str]:
    """Return only documented operator variables plus declared overrides."""

    host = os.environ if source is None else source
    allowed = _OPERATOR_ENVIRONMENT_ALLOWLIST | frozenset(include)
    environment = {name: host[name] for name in allowed if name in host}
    if declared is not None:
        environment.update(declared)
    return MappingProxyType(environment)


def _read_stream(
    stream: object,
    target: bytearray,
    limit: int,
    exceeded: threading.Event,
    sink: Callable[[bytes], None] | None,
    sink_name: str,
    sink_failure: queue.Queue[tuple[str, str]],
) -> None:
    read = getattr(stream, "read1", None) or getattr(stream, "read", None)
    assert read is not None
    while True:
        block = read(65_536)
        if not block:
            return
        remaining = max(0, limit - len(target))
        accepted = block[:remaining]
        target.extend(accepted)
        if accepted and sink is not None:
            try:
                sink(bytes(accepted))
            except Exception as exc:
                with suppress(queue.Full):
                    sink_failure.put_nowait((sink_name, type(exc).__name__))
                return
        if len(block) > remaining:
            exceeded.set()
            return


def _launch_tool_process(
    request: ToolCommandRequest,
) -> subprocess.Popen[bytes] | tuple[None, str]:
    try:
        return subprocess.Popen(
            [request.executable, *request.arguments],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=request.cwd,
            env=dict(request.environment),
            start_new_session=os.name == "posix",
            creationflags=(
                int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
                if os.name == "nt"
                else 0
            ),
        )
    except OSError as exc:
        return None, type(exc).__name__


def _kill_tool_process_tree(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    else:  # pragma: no cover - exercised in remote Windows validation
        if _TASKKILL is not None:
            cleanup = subprocess.Popen(
                [_TASKKILL, "/T", "/F", "/PID", str(process.pid)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=dict(operator_environment()),
            )
            with suppress(subprocess.TimeoutExpired):
                cleanup.wait(timeout=5.0)
                return
            cleanup.kill()
        else:
            process.kill()


def _poll_tool_status(
    process: subprocess.Popen[bytes],
    request: ToolCommandRequest,
    stdout_overflow: threading.Event,
    stderr_overflow: threading.Event,
    sink_failure: queue.Queue[tuple[str, str]],
    deadline: float,
) -> ToolCommandStatus:
    while process.poll() is None:
        if (
            request.cancellation_event is not None
            and request.cancellation_event.is_set()
        ):
            return ToolCommandStatus.CANCELLED
        if stdout_overflow.is_set() or stderr_overflow.is_set():
            return ToolCommandStatus.OUTPUT_LIMIT_EXCEEDED
        if not sink_failure.empty():
            return ToolCommandStatus.STREAM_FAILED
        if time.monotonic() >= deadline:
            return ToolCommandStatus.TIMED_OUT
        time.sleep(0.01)
    return ToolCommandStatus.EXITED


def run_tool_command(
    request: ToolCommandRequest,
) -> ToolCommandResult:
    """Execute a tooling request without shell parsing or ambient discovery."""

    process = _launch_tool_process(request)
    if isinstance(process, tuple):
        return ToolCommandResult(
            status=ToolCommandStatus.START_FAILED,
            exit_code=None,
            stdout=b"",
            stderr=b"",
            diagnostic=process[1],
        )
    stdout, stderr, stdout_overflow, stderr_overflow, sink_failure, readers = (
        _start_stream_readers(process, request)
    )
    _start_input_writer(process, request.stdin_bytes)
    deadline = time.monotonic() + request.timeout_seconds
    status = _poll_tool_status(
        process,
        request,
        stdout_overflow,
        stderr_overflow,
        sink_failure,
        deadline,
    )
    return _finish_tool_process(
        process,
        status=status,
        readers=readers,
        stdout=stdout,
        stderr=stderr,
        stdout_overflow=stdout_overflow,
        stderr_overflow=stderr_overflow,
        sink_failure=sink_failure,
    )


def _start_stream_readers(
    process: subprocess.Popen[bytes], request: ToolCommandRequest
) -> tuple[
    bytearray,
    bytearray,
    threading.Event,
    threading.Event,
    queue.Queue[tuple[str, str]],
    tuple[threading.Thread, threading.Thread],
]:
    stdout, stderr = bytearray(), bytearray()
    stdout_overflow, stderr_overflow = threading.Event(), threading.Event()
    sink_failure: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=1)
    readers = (
        threading.Thread(
            target=_read_stream,
            args=(
                process.stdout,
                stdout,
                request.stdout_limit_bytes,
                stdout_overflow,
                request.stdout_sink,
                "stdout",
                sink_failure,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=_read_stream,
            args=(
                process.stderr,
                stderr,
                request.stderr_limit_bytes,
                stderr_overflow,
                request.stderr_sink,
                "stderr",
                sink_failure,
            ),
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()
    return stdout, stderr, stdout_overflow, stderr_overflow, sink_failure, readers


def _start_input_writer(process: subprocess.Popen[bytes], stdin_bytes: bytes) -> None:
    def write_input() -> None:
        try:
            assert process.stdin is not None
            process.stdin.write(stdin_bytes)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    threading.Thread(target=write_input, daemon=True).start()


def _finish_tool_process(
    process: subprocess.Popen[bytes],
    *,
    status: ToolCommandStatus,
    readers: tuple[threading.Thread, threading.Thread],
    stdout: bytearray,
    stderr: bytearray,
    stdout_overflow: threading.Event,
    stderr_overflow: threading.Event,
    sink_failure: queue.Queue[tuple[str, str]],
) -> ToolCommandResult:
    if status is not ToolCommandStatus.EXITED:
        _kill_tool_process_tree(process)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            return ToolCommandResult(
                status=status,
                exit_code=None,
                stdout=bytes(stdout),
                stderr=bytes(stderr),
                stdout_exceeded=stdout_overflow.is_set(),
                stderr_exceeded=stderr_overflow.is_set(),
                diagnostic="tool process did not stop within the shutdown deadline",
            )
    for reader in readers:
        reader.join(timeout=1.0)
    if status is ToolCommandStatus.EXITED:
        if stdout_overflow.is_set() or stderr_overflow.is_set():
            status = ToolCommandStatus.OUTPUT_LIMIT_EXCEEDED
        elif not sink_failure.empty():
            status = ToolCommandStatus.STREAM_FAILED
    diagnostic = None
    if status is ToolCommandStatus.STREAM_FAILED:
        sink_name, exception_name = sink_failure.get_nowait()
        diagnostic = f"{sink_name} sink failed: {exception_name}"
    return ToolCommandResult(
        status=status,
        exit_code=process.returncode if status is ToolCommandStatus.EXITED else None,
        stdout=bytes(stdout),
        stderr=bytes(stderr),
        stdout_exceeded=stdout_overflow.is_set(),
        stderr_exceeded=stderr_overflow.is_set(),
        diagnostic=diagnostic,
    )


class ToolInteractiveStatus(StrEnum):
    """Terminal state of one interactive tooling process."""

    STARTED = "STARTED"
    START_FAILED = "START_FAILED"
    CLOSED = "CLOSED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class ToolInteractiveRequest:
    """Immutable request for one long-lived interactive tooling process.

    The process owns its own process group, bounded stderr capture, startup
    deadline, per-read deadline, and shutdown deadline.  Callers exchange
    line-delimited request/response frames through the typed
    :class:`ToolInteractiveCommand` methods and never touch ``subprocess``
    directly.
    """

    executable: str
    arguments: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    cwd: str = ""
    startup_timeout_seconds: float = 0.0
    read_timeout_seconds: float = 0.0
    shutdown_timeout_seconds: float = 5.0
    stdout_limit_bytes: int = 256 * 1024
    stderr_limit_bytes: int = 256 * 1024
    cancellation_event: threading.Event | None = None

    def __post_init__(self) -> None:
        self._validate_command()
        self._validate_io_policy()

    def _validate_command(self) -> None:
        if not Path(self.executable).is_absolute():
            raise ValueError("interactive executable must be an absolute path")
        object.__setattr__(self, "arguments", tuple(self.arguments))
        if not all(isinstance(argument, str) for argument in self.arguments):
            raise TypeError("interactive arguments must be strings")
        if not isinstance(self.environment, Mapping):
            raise TypeError("interactive environment must be a mapping")
        environment = dict(self.environment)
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        ):
            raise TypeError("interactive environment keys and values must be strings")
        object.__setattr__(self, "environment", MappingProxyType(environment))
        if not self.cwd or not Path(self.cwd).is_absolute():
            raise ValueError(
                "interactive working directory must be an explicit absolute path"
            )

    def _validate_io_policy(self) -> None:
        for name in ("startup_timeout_seconds", "read_timeout_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            not math.isfinite(self.shutdown_timeout_seconds)
            or self.shutdown_timeout_seconds <= 0
        ):
            raise ValueError("shutdown_timeout_seconds must be positive and finite")
        for name in ("stdout_limit_bytes", "stderr_limit_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.cancellation_event is not None and not isinstance(
            self.cancellation_event, threading.Event
        ):
            raise TypeError("cancellation_event must be a threading.Event or None")


class ToolInteractiveCommand:
    """A narrowly typed long-lived child for request/response protocols.

    Owns process creation in its own process group, bounded stderr capture,
    startup/read/shutdown deadlines, and process-group cleanup.  Callers
    exchange line-delimited request/response frames through :meth:`send` and
    :meth:`read_response` and never touch ``subprocess`` directly.

    A dedicated stdout reader thread feeds lines into a queue so the
    deadline loop can poll with bounded intervals instead of blocking on
    ``readline()``.
    """

    def __init__(self, request: ToolInteractiveRequest) -> None:
        self._request = request
        self._process: subprocess.Popen[str] | None = None
        self._stderr = bytearray()
        self._stderr_thread: threading.Thread | None = None
        self._stderr_exceeded = threading.Event()
        self._stdout_queue: queue.Queue[str | None] = queue.Queue(maxsize=1024)
        self._stdout_thread: threading.Thread | None = None
        self._stdout_closed = threading.Event()
        self._stdout_exceeded = threading.Event()
        self._responses_read = 0
        self._status = ToolInteractiveStatus.START_FAILED

    @property
    def status(self) -> ToolInteractiveStatus:
        return self._status

    @property
    def stderr(self) -> bytes:
        return bytes(self._stderr)

    @property
    def stderr_exceeded(self) -> bool:
        return self._stderr_exceeded.is_set()

    def start(self) -> None:
        """Launch the child and start background reader threads."""
        try:
            self._process = subprocess.Popen(
                [self._request.executable, *self._request.arguments],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._request.cwd,
                env=dict(self._request.environment),
                text=True,
                bufsize=1,
                start_new_session=os.name == "posix",
                creationflags=(
                    int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
                    if os.name == "nt"
                    else 0
                ),
            )
        except OSError:
            self._status = ToolInteractiveStatus.START_FAILED
            raise
        self._status = ToolInteractiveStatus.STARTED
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._stdout_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._stdout_thread.start()

    def _drain_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        while True:
            block = self._process.stderr.read(4096)
            if not block:
                return
            encoded = block.encode("utf-8", "replace")
            remaining = max(0, self._request.stderr_limit_bytes - len(self._stderr))
            self._stderr.extend(encoded[:remaining])
            if len(encoded) > remaining:
                self._stderr_exceeded.set()

    def _drain_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        while True:
            line = self._process.stdout.readline(self._request.stdout_limit_bytes + 1)
            if line == "":
                self._stdout_closed.set()
                self._stdout_queue.put(None)
                return
            if len(line.encode("utf-8", "replace")) > self._request.stdout_limit_bytes:
                self._stdout_exceeded.set()
                return
            self._stdout_queue.put(line)

    def send(self, frame: str) -> None:
        """Write one request frame (without trailing newline) to the child stdin."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("interactive process is not started")
        try:
            self._process.stdin.write(frame + "\n\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._abort(ToolInteractiveStatus.CLOSED)
            raise RuntimeError(
                "interactive process closed before receiving input"
            ) from exc

    def _check_read_abort(self, deadline: float) -> Exception | None:
        """Return an abort exception if a read must stop, else ``None``."""

        if (
            self._request.cancellation_event is not None
            and self._request.cancellation_event.is_set()
        ):
            self._abort(ToolInteractiveStatus.CANCELLED)
            return TimeoutError("interactive process was cancelled")
        if self._stderr_exceeded.is_set():
            self._abort(ToolInteractiveStatus.CLOSED)
            return RuntimeError("interactive process exceeded stderr bounds")
        if self._stdout_exceeded.is_set():
            self._abort(ToolInteractiveStatus.CLOSED)
            return RuntimeError("interactive process exceeded stdout bounds")
        if time.monotonic() >= deadline:
            self._abort(ToolInteractiveStatus.TIMED_OUT)
            return TimeoutError("interactive read timed out")
        return None

    def read_response(self) -> str:
        """Read one blank-line-terminated response frame from the child stdout."""
        if self._process is None:
            raise RuntimeError("interactive process is not started")
        lines: list[str] = []
        timeout = (
            self._request.startup_timeout_seconds
            if self._responses_read == 0
            else self._request.read_timeout_seconds
        )
        deadline = time.monotonic() + timeout
        response_bytes = 0
        while True:
            abort = self._check_read_abort(deadline)
            if abort is not None:
                raise abort
            try:
                item = self._stdout_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                raise RuntimeError("interactive process closed before responding")
            if not item.strip():
                if lines:
                    # Re-check stream bounds before accepting a completed frame.
                    abort = self._check_read_abort(deadline)
                    if abort is not None:
                        raise abort
                    self._responses_read += 1
                    return "\n".join(lines)
                continue
            response_bytes += len(item.encode("utf-8", "replace"))
            if response_bytes > self._request.stdout_limit_bytes:
                self._abort(ToolInteractiveStatus.CLOSED)
                raise RuntimeError("interactive process exceeded stdout bounds")
            lines.append(item.rstrip("\n"))

    def _abort(self, status: ToolInteractiveStatus) -> None:
        self._status = status
        if self._process is None or self._process.poll() is not None:
            return
        self._terminate_group()
        with suppress(subprocess.TimeoutExpired):
            self._process.wait(timeout=self._request.shutdown_timeout_seconds)

    def close(self) -> int:
        """Close stdin, wait within the shutdown deadline, and return the exit code."""
        if self._process is None:
            self._status = ToolInteractiveStatus.CLOSED
            return -1
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            return_code = self._process.wait(
                timeout=self._request.shutdown_timeout_seconds
            )
        except subprocess.TimeoutExpired:
            self._terminate_group()
            try:
                return_code = self._process.wait(
                    timeout=self._request.shutdown_timeout_seconds
                )
            except subprocess.TimeoutExpired:
                return_code = -1
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=1.0)
        self._status = ToolInteractiveStatus.CLOSED
        return return_code if return_code is not None else -1

    def _terminate_group(self) -> None:
        assert self._process is not None
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(self._process.pid, signal.SIGKILL)
        else:  # pragma: no cover - exercised in remote Windows validation
            if _TASKKILL is not None:
                cleanup = subprocess.Popen(
                    [_TASKKILL, "/T", "/F", "/PID", str(self._process.pid)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=dict(operator_environment()),
                )
                try:
                    cleanup.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    cleanup.kill()
            else:
                self._process.kill()
