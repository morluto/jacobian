"""Bounded subprocess execution for concrete external operations.

One-shot children run in their own process group. Reader threads cap retained
output and terminate the whole group as soon as either stream exceeds its
limit. Timeouts also terminate descendants rather than only the immediate
worker.

The ordinary gateway is a small low-level primitive for genuinely isolated
one-shot commands. A second, narrower gateway supports one adaptive exchange
from inside an already supervised worker. That nested child inherits the
worker's process group, is visible only through one callback-scoped dialogue,
and is never exposed as a reusable process session.

Each domain operation owns its command, temporary files, input, dialogue, and
conversion to a typed mathematical result.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, Never, cast

from jacobian._execution import (
    RequestCancellationSignal,
    current_request_cancellation,
    request_cancellation,
)

__all__ = [
    "BoundedProcessResult",
    "BoundedWorkerDialogue",
    "BoundedWorkerDialogueCompleted",
    "BoundedWorkerDialogueError",
    "BoundedWorkerDialogueErrorReason",
    "ProcessPlatformTools",
    "ProcessResourceLimits",
    "bounded_process_cancellation",
    "run_bounded_process",
    "run_bounded_worker_dialogue",
    "worker_environment",
]


_PIPE_DRAIN_GRACE_SECONDS = 0.5
_DEFAULT_LOCALE = "C.UTF-8"


@dataclass(frozen=True, slots=True)
class BoundedProcessResult:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    stdout_exceeded: bool
    stderr_exceeded: bool
    timed_out: bool
    cancelled: bool = False


class BoundedWorkerDialogueErrorReason(StrEnum):
    """Operational reason one callback-scoped worker dialogue could not finish."""

    START_FAILED = "START_FAILED"
    DEADLINE_EXPIRED = "DEADLINE_EXPIRED"
    STDOUT_LIMIT = "STDOUT_LIMIT"
    STDERR_LIMIT = "STDERR_LIMIT"
    CLOSED = "CLOSED"
    NONZERO_EXIT = "NONZERO_EXIT"


class BoundedWorkerDialogueError(RuntimeError):
    """Typed failure from :func:`run_bounded_worker_dialogue`."""

    reason: BoundedWorkerDialogueErrorReason
    stderr: bytes

    def __init__(
        self,
        reason: BoundedWorkerDialogueErrorReason,
        *,
        stderr: bytes,
    ) -> None:
        self.reason = reason
        self.stderr = stderr
        super().__init__(reason.value)


@dataclass(frozen=True, slots=True)
class BoundedWorkerDialogueCompleted[ValueT]:
    """Value returned by a dialogue callback and its cumulative stdout use."""

    value: ValueT
    stdout_bytes: int


class _WorkerDialogueControlError(Exception):
    def __init__(self, reason: BoundedWorkerDialogueErrorReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


@dataclass(slots=True)
class _DialogueWrite:
    done: bool = False
    error: BaseException | None = None


class _BoundedWorkerDialogueState:
    """Process-owned state hidden behind the callback-scoped public surface."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        execution_deadline: float,
        absolute_deadline: float,
        stdout_limit: int,
        stderr_limit: int,
    ) -> None:
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        self._process = process
        self._stdin = process.stdin
        self._stdout = process.stdout
        self._stderr = process.stderr
        self._execution_deadline = execution_deadline
        self._absolute_deadline = absolute_deadline
        self._stdout_limit = stdout_limit
        self._stderr_limit = stderr_limit
        self._condition = threading.Condition()
        self._stdout_buffer = bytearray()
        self._stderr_buffer = bytearray()
        self._stdout_bytes = 0
        self._stderr_bytes = 0
        self._stdout_closed = False
        self._failure: BoundedWorkerDialogueErrorReason | None = None
        self._active = True
        self._writers: list[threading.Thread] = []
        self._readers = (
            threading.Thread(
                target=self._capture_stdout,
                name="bounded-worker-dialogue-stdout",
                daemon=True,
            ),
            threading.Thread(
                target=self._capture_stderr,
                name="bounded-worker-dialogue-stderr",
                daemon=True,
            ),
        )

    @property
    def stderr(self) -> bytes:
        with self._condition:
            return bytes(self._stderr_buffer)

    @property
    def stdout_bytes(self) -> int:
        with self._condition:
            return self._stdout_bytes

    def start(self) -> None:
        for reader in self._readers:
            reader.start()

    def deactivate(self) -> None:
        with self._condition:
            self._active = False
            self._condition.notify_all()

    def _require_active_locked(self) -> None:
        if not self._active:
            raise RuntimeError("worker dialogue scope has ended")

    def _record_failure_locked(self, reason: BoundedWorkerDialogueErrorReason) -> None:
        if self._failure is None:
            self._failure = reason
        self._condition.notify_all()

    def _kill_immediate_child(self) -> None:
        if self._process.poll() is None:
            with suppress(OSError):
                self._process.kill()

    def _fail(self, reason: BoundedWorkerDialogueErrorReason) -> Never:
        with self._condition:
            self._record_failure_locked(reason)
        self._kill_immediate_child()
        raise _WorkerDialogueControlError(reason)

    def _raise_recorded_failure_locked(self) -> None:
        if self._failure is not None:
            raise _WorkerDialogueControlError(self._failure)

    def _capture_stdout(self) -> None:
        self._capture_stream(stdout=True)

    def _capture_stderr(self) -> None:
        self._capture_stream(stdout=False)

    def _capture_stream(self, *, stdout: bool) -> None:
        stream = self._stdout if stdout else self._stderr
        limit = self._stdout_limit if stdout else self._stderr_limit
        reason = (
            BoundedWorkerDialogueErrorReason.STDOUT_LIMIT
            if stdout
            else BoundedWorkerDialogueErrorReason.STDERR_LIMIT
        )
        try:
            while chunk := stream.read(64 * 1024):
                exceeded = False
                with self._condition:
                    previous = self._stdout_bytes if stdout else self._stderr_bytes
                    accepted = max(0, limit - previous)
                    if stdout:
                        self._stdout_buffer.extend(chunk[:accepted])
                        self._stdout_bytes = previous + len(chunk)
                    else:
                        self._stderr_buffer.extend(chunk[:accepted])
                        self._stderr_bytes = previous + len(chunk)
                    exceeded = previous + len(chunk) > limit
                    if exceeded:
                        self._record_failure_locked(reason)
                    else:
                        self._condition.notify_all()
                if exceeded:
                    self._kill_immediate_child()
                    return
        except (OSError, ValueError):
            pass
        finally:
            with self._condition:
                if stdout:
                    self._stdout_closed = True
                self._condition.notify_all()

    def _closed_reason(self) -> BoundedWorkerDialogueErrorReason:
        returncode = self._process.poll()
        if returncode is not None and returncode != 0:
            return BoundedWorkerDialogueErrorReason.NONZERO_EXIT
        return BoundedWorkerDialogueErrorReason.CLOSED

    def send(self, payload: bytes) -> None:
        with self._condition:
            self._require_active_locked()
            self._raise_recorded_failure_locked()
            deadline_expired = time.monotonic() >= self._execution_deadline
        if deadline_expired:
            self._fail(BoundedWorkerDialogueErrorReason.DEADLINE_EXPIRED)
        if not payload:
            return

        write = _DialogueWrite()

        def write_payload() -> None:
            try:
                remaining = memoryview(payload)
                while remaining:
                    written = self._stdin.write(remaining)
                    if written is None or written <= 0:
                        raise BrokenPipeError("worker stdin closed")
                    remaining = remaining[written:]
            except BaseException as exc:  # recorded for the coordinating thread
                write.error = exc
            finally:
                with self._condition:
                    write.done = True
                    self._condition.notify_all()

        writer = threading.Thread(
            target=write_payload,
            name="bounded-worker-dialogue-stdin",
            daemon=True,
        )
        self._writers.append(writer)
        writer.start()

        while True:
            with self._condition:
                self._require_active_locked()
                self._raise_recorded_failure_locked()
                deadline_expired = time.monotonic() >= self._execution_deadline
                if deadline_expired:
                    pass
                elif write.done:
                    if write.error is not None:
                        reason = self._closed_reason()
                        self._record_failure_locked(reason)
                        raise _WorkerDialogueControlError(reason) from write.error
                    return
                remaining_seconds = self._execution_deadline - time.monotonic()
                if remaining_seconds > 0:
                    self._condition.wait(timeout=remaining_seconds)
                    continue
            self._fail(BoundedWorkerDialogueErrorReason.DEADLINE_EXPIRED)

    def read_until(self, marker: bytes, *, frame_limit: int) -> bytes:
        if not marker:
            raise ValueError("worker dialogue marker must not be empty")
        if frame_limit <= 0:
            raise ValueError("worker dialogue frame limit must be positive")
        if len(marker) > frame_limit:
            raise ValueError("worker dialogue marker exceeds the frame limit")

        while True:
            with self._condition:
                self._require_active_locked()
                self._raise_recorded_failure_locked()
                deadline_expired = time.monotonic() >= self._execution_deadline
                marker_start = self._stdout_buffer.find(marker)
                if deadline_expired:
                    self._record_failure_locked(
                        BoundedWorkerDialogueErrorReason.DEADLINE_EXPIRED
                    )
                elif marker_start >= 0:
                    frame_end = marker_start + len(marker)
                    if frame_end <= frame_limit:
                        frame = bytes(self._stdout_buffer[:frame_end])
                        del self._stdout_buffer[:frame_end]
                        return frame
                    self._record_failure_locked(
                        BoundedWorkerDialogueErrorReason.STDOUT_LIMIT
                    )
                elif len(self._stdout_buffer) < frame_limit:
                    if self._stdout_closed:
                        reason = self._closed_reason()
                        self._record_failure_locked(reason)
                    else:
                        remaining_seconds = self._execution_deadline - time.monotonic()
                        if remaining_seconds > 0:
                            self._condition.wait(timeout=remaining_seconds)
                            continue
                        self._record_failure_locked(
                            BoundedWorkerDialogueErrorReason.DEADLINE_EXPIRED
                        )
                else:
                    self._record_failure_locked(
                        BoundedWorkerDialogueErrorReason.STDOUT_LIMIT
                    )
                assert self._failure is not None
                reason = self._failure
            self._kill_immediate_child()
            raise _WorkerDialogueControlError(reason)

    def finish(self) -> None:
        with self._condition:
            self._raise_recorded_failure_locked()
        if time.monotonic() >= self._execution_deadline:
            self._fail(BoundedWorkerDialogueErrorReason.DEADLINE_EXPIRED)

        # Pipes are unbuffered, so closing stdin cannot acquire an implicit
        # flush budget. EOF lets command-line tools finish after the callback.
        with suppress(OSError):
            self._stdin.close()
        remaining_seconds = self._execution_deadline - time.monotonic()
        if remaining_seconds <= 0:
            self._fail(BoundedWorkerDialogueErrorReason.DEADLINE_EXPIRED)
        try:
            self._process.wait(timeout=remaining_seconds)
        except subprocess.TimeoutExpired:
            self._fail(BoundedWorkerDialogueErrorReason.DEADLINE_EXPIRED)

        for reader in self._readers:
            reader.join(timeout=max(0.0, self._absolute_deadline - time.monotonic()))
        with self._condition:
            self._raise_recorded_failure_locked()
        if any(reader.is_alive() for reader in self._readers):
            self._fail(BoundedWorkerDialogueErrorReason.DEADLINE_EXPIRED)
        if self._process.returncode != 0:
            raise _WorkerDialogueControlError(
                BoundedWorkerDialogueErrorReason.NONZERO_EXIT
            )

    def cleanup(self, *, kill: bool) -> None:
        if kill:
            self._kill_immediate_child()
        with suppress(OSError):
            self._stdin.close()
        remaining_seconds = max(0.0, self._absolute_deadline - time.monotonic())
        try:
            self._process.wait(timeout=remaining_seconds)
        except subprocess.TimeoutExpired:
            self._kill_immediate_child()
            with suppress(subprocess.TimeoutExpired):
                self._process.wait(
                    timeout=max(0.0, self._absolute_deadline - time.monotonic())
                )
        threads = (*self._readers, *self._writers)
        for thread in threads:
            if thread.ident is not None:
                thread.join(
                    timeout=max(0.0, self._absolute_deadline - time.monotonic())
                )
        for stream in (self._stdout, self._stderr):
            with suppress(OSError):
                stream.close()
        for thread in threads:
            if thread.ident is not None:
                thread.join(
                    timeout=max(0.0, self._absolute_deadline - time.monotonic())
                )


class BoundedWorkerDialogue:
    """Callback-scoped byte dialogue with one already supervised child."""

    __slots__ = ("__state",)

    def __init__(self, state: _BoundedWorkerDialogueState) -> None:
        self.__state = state

    def send(self, payload: bytes) -> None:
        """Write all *payload* before the operation's absolute deadline."""

        self.__state.send(payload)

    def read_until(self, marker: bytes, *, frame_limit: int) -> bytes:
        """Read through *marker*, preserving any later bytes for the next frame."""

        return self.__state.read_until(marker, frame_limit=frame_limit)


@contextmanager
def bounded_process_cancellation(
    event: RequestCancellationSignal,
) -> Iterator[None]:
    """Bind cooperative subprocess cancellation to the current worker context."""

    with request_cancellation(event):
        yield


@dataclass(frozen=True, slots=True)
class ProcessResourceLimits:
    """Portable subset of operating-system worker resource limits.

    Limits are applied on POSIX platforms that expose ``resource.prlimit``.
    Other platforms retain the existing wall-time and output limits.  A limit
    is *active* only when its field is not ``None``; the engine skips prlimit
    wrapping entirely when no field is active.
    """

    cpu_seconds: int | None = None
    address_space_bytes: int | None = None
    file_size_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.cpu_seconds is not None and self.cpu_seconds <= 0:
            raise ValueError("CPU limit must be positive")
        if self.address_space_bytes is not None and self.address_space_bytes <= 0:
            raise ValueError("address-space limit must be positive")
        if self.file_size_bytes is not None and self.file_size_bytes <= 0:
            raise ValueError("file-size limit must be positive")

    def has_active_limit(self) -> bool:
        return (
            self.cpu_seconds is not None
            or self.address_space_bytes is not None
            or self.file_size_bytes is not None
        )


@dataclass(frozen=True, slots=True)
class ProcessPlatformTools:
    """Bootstrap-resolved absolute paths to platform helper executables.

    The engine consumes these directly and never performs executable
    discovery when they are provided.  Use :meth:`discover` at bootstrap to
    resolve the helpers available on the current platform; fields are ``None``
    when the helper is absent or the platform does not use it.  On Windows
    ``taskkill_executable`` must be an absolute system helper resolved at
    bootstrap; the engine does not claim an ambient fallback when this field
    is supplied.
    """

    prlimit_executable: str | None = None
    taskkill_executable: str | None = None

    def __post_init__(self) -> None:
        for name in ("prlimit_executable", "taskkill_executable"):
            value = getattr(self, name)
            if value is not None and not Path(value).is_absolute():
                raise ValueError(f"{name} must be an absolute path")


def _apply_resource_limits(
    process: subprocess.Popen[bytes],
    limits: ProcessResourceLimits,
) -> None:
    """Apply supported hard limits before accepting worker output."""

    if os.name != "posix":
        return
    try:
        import resource
    except ImportError:  # pragma: no cover - platform dependent
        return
    candidate = getattr(resource, "prlimit", None)
    if not callable(candidate):  # pragma: no cover - platform dependent
        return
    prlimit = cast(Callable[[int, int, tuple[int, int]], object], candidate)

    def set_limit(kind: int, value: int) -> None:
        # A very short-lived child may exit between Popen and prlimit. It can
        # no longer consume resources, so there is nothing left to constrain.
        with suppress(ProcessLookupError):
            prlimit(process.pid, kind, (value, value))

    if limits.cpu_seconds is not None:
        set_limit(resource.RLIMIT_CPU, limits.cpu_seconds)
    if limits.address_space_bytes is not None:
        set_limit(resource.RLIMIT_AS, limits.address_space_bytes)
    if limits.file_size_bytes is not None:
        set_limit(resource.RLIMIT_FSIZE, limits.file_size_bytes)


def _resource_limited_command(
    command: Sequence[str],
    limits: ProcessResourceLimits,
    prlimit_executable: str | None,
) -> tuple[list[str], bool]:
    """Use util-linux prlimit so limits are installed before target execution.

    Never wraps when *limits* has no active field or when bootstrap did not
    supply a resolved prlimit executable.
    """

    if not limits.has_active_limit():
        return list(command), False
    if prlimit_executable is None:
        return list(command), False
    prlimit = prlimit_executable
    options: list[str] = []
    if limits.cpu_seconds is not None:
        options.append(f"--cpu={limits.cpu_seconds}:{limits.cpu_seconds}")
    if limits.address_space_bytes is not None:
        options.append(
            f"--as={limits.address_space_bytes}:{limits.address_space_bytes}"
        )
    if limits.file_size_bytes is not None:
        options.append(f"--fsize={limits.file_size_bytes}:{limits.file_size_bytes}")
    return [prlimit, *options, "--", *command], True


def _kill_process_tree(
    process: subprocess.Popen[bytes],
    platform_tools: ProcessPlatformTools | None = None,
) -> None:
    """Best-effort termination of a worker and every descendant it created."""

    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        return

    taskkill = (
        platform_tools.taskkill_executable if platform_tools is not None else None
    )
    if taskkill is not None:  # pragma: no cover - exercised in cross-platform CI
        subprocess.run(
            [taskkill, "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
        )
        return

    process.kill()  # pragma: no cover - defensive fallback


def _capture_stream(
    stream: BinaryIO,
    *,
    limit: int,
    process: subprocess.Popen[bytes],
    captured: bytearray,
    exceeded: threading.Event,
    platform_tools: ProcessPlatformTools | None = None,
) -> None:
    total = 0
    read_chunk = getattr(stream, "read1", stream.read)
    try:
        while chunk := read_chunk(64 * 1024):
            total += len(chunk)
            remaining = max(0, limit - len(captured))
            captured.extend(chunk[:remaining])
            if total > limit:
                exceeded.set()
                _kill_process_tree(process, platform_tools)
                return
    except (OSError, ValueError):
        # The coordinator may close the pipe after killing descendants that
        # retained it beyond the operation deadline.
        return
    finally:
        stream.close()


def _apply_post_start_limits(
    process: subprocess.Popen[bytes],
    resource_limits: ProcessResourceLimits | None,
    limits_applied_before_exec: bool,
    platform_tools: ProcessPlatformTools | None,
) -> None:
    """Apply resource limits after exec when prlimit wrapping was not used."""

    if (
        resource_limits is None
        or not resource_limits.has_active_limit()
        or limits_applied_before_exec
    ):
        return
    try:
        _apply_resource_limits(process, resource_limits)
    except (OSError, ValueError):
        _kill_process_tree(process, platform_tools)
        process.wait()
        raise


def _monitor_bounded_process(
    process: subprocess.Popen[bytes],
    *,
    deadline: float,
    cancellation_event: RequestCancellationSignal | None,
    platform_tools: ProcessPlatformTools | None,
) -> tuple[bool, bool]:
    """Poll the child until exit, timeout, or cancellation.

    Returns ``(timed_out, cancelled)``.
    """

    # Setup may consume the execution allowance before monitoring starts.  Do
    # not accept an already-exited child as timely merely because ``poll``
    # would skip the loop below.
    timed_out = time.monotonic() >= deadline
    cancelled = False
    if timed_out:
        _kill_process_tree(process, platform_tools)
        process.wait()
        return timed_out, cancelled
    while process.poll() is None:
        if cancellation_event is not None and cancellation_event.is_set():
            cancelled = True
            _kill_process_tree(process, platform_tools)
            process.wait()
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            _kill_process_tree(process, platform_tools)
            process.wait()
            break
        try:
            process.wait(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            continue
    return timed_out, cancelled


def _drain_reader_threads(
    process: subprocess.Popen[bytes],
    readers: tuple[threading.Thread, ...],
    platform_tools: ProcessPlatformTools | None,
    *,
    cleanup_deadline: float,
) -> bool:
    """Kill the tree, drain readers, and return ``True`` if any survived."""

    _kill_process_tree(process, platform_tools)
    for reader in readers:
        reader.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
    return any(reader.is_alive() for reader in readers)


def run_bounded_process(
    command: Sequence[str],
    *,
    input_bytes: bytes,
    timeout_seconds: float,
    environment: Mapping[str, str],
    stdout_limit: int,
    stderr_limit: int,
    resource_limits: ProcessResourceLimits | None = None,
    cwd: str | None = None,
    platform_tools: ProcessPlatformTools | None = None,
    cancellation_event: RequestCancellationSignal | None = None,
) -> BoundedProcessResult:
    """Run a child with bounded output, time, lifetime, and supported resources.

    *cwd* is an optional absolute working directory passed to the child; when
    ``None`` the child inherits the engine's cwd.  *platform_tools* carries
    bootstrap-resolved absolute helper paths (prlimit, taskkill); the engine
    never discovers executables.  *cancellation_event* overrides the context-bound cancellation
    event for explicit callers; when ``None`` the engine uses the
    context-bound event set by :func:`bounded_process_cancellation`.

    Commands using ``sys.executable`` receive the loaded package's import root
    when *environment* does not explicitly supply ``PYTHONPATH``.
    """

    if stdout_limit < 0 or stderr_limit < 0:
        raise ValueError("subprocess output limits must be nonnegative")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("subprocess timeout must be positive")

    # This envelope includes input spooling, process setup, execution, result
    # capture, and reaping.  Keep a finite portion for teardown so a timeout
    # cannot acquire a second, fresh cleanup clock after the admitted lifetime.
    started = time.monotonic()
    absolute_deadline = started + timeout_seconds
    cleanup_allowance = min(_PIPE_DRAIN_GRACE_SECONDS, timeout_seconds / 100)
    execution_deadline = absolute_deadline - cleanup_allowance

    start_new_session = os.name == "posix"
    creationflags = (
        int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        if os.name == "nt"
        else 0
    )

    stdout = bytearray()
    stderr = bytearray()
    stdout_exceeded = threading.Event()
    stderr_exceeded = threading.Event()
    if cancellation_event is None:
        cancellation_event = current_request_cancellation()

    prlimit_executable = (
        platform_tools.prlimit_executable if platform_tools is not None else None
    )

    with tempfile.TemporaryFile() as stdin_file:
        stdin_file.write(input_bytes)
        stdin_file.seek(0)
        # Spooling is part of the admitted execution envelope.  Do not launch
        # a worker after it has consumed the request's complete deadline.
        if cancellation_event is not None and cancellation_event.is_set():
            return BoundedProcessResult(
                returncode=None,
                stdout=b"",
                stderr=b"",
                stdout_exceeded=False,
                stderr_exceeded=False,
                timed_out=False,
                cancelled=True,
            )
        if time.monotonic() >= execution_deadline:
            return BoundedProcessResult(
                returncode=None,
                stdout=b"",
                stderr=b"",
                stdout_exceeded=False,
                stderr_exceeded=False,
                timed_out=True,
            )
        bounded_command, limits_applied_before_exec = (
            _resource_limited_command(command, resource_limits, prlimit_executable)
            if resource_limits is not None
            else (list(command), False)
        )
        process = subprocess.Popen(
            bounded_command,
            stdin=stdin_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_command_environment(command, environment),
            cwd=cwd,
            start_new_session=start_new_session,
            creationflags=creationflags,
        )
        try:
            _apply_post_start_limits(
                process, resource_limits, limits_applied_before_exec, platform_tools
            )
            assert process.stdout is not None
            assert process.stderr is not None
        except BaseException:
            # _apply_post_start_limits already killed and waited the
            # process on failure, but the pipes were never closed.  Close
            # them to avoid leaking file descriptors.
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            raise

        readers = (
            threading.Thread(
                target=_capture_stream,
                kwargs={
                    "stream": process.stdout,
                    "limit": stdout_limit,
                    "process": process,
                    "captured": stdout,
                    "exceeded": stdout_exceeded,
                    "platform_tools": platform_tools,
                },
                daemon=True,
            ),
            threading.Thread(
                target=_capture_stream,
                kwargs={
                    "stream": process.stderr,
                    "limit": stderr_limit,
                    "process": process,
                    "captured": stderr,
                    "exceeded": stderr_exceeded,
                    "platform_tools": platform_tools,
                },
                daemon=True,
            ),
        )
        for reader in readers:
            reader.start()

        timed_out = False
        cancelled = False
        try:
            timed_out, cancelled = _monitor_bounded_process(
                process,
                deadline=execution_deadline,
                cancellation_event=cancellation_event,
                platform_tools=platform_tools,
            )
        finally:
            # A clean worker may leave descendants holding inherited pipe
            # handles. Terminate the process group before draining so those
            # handles cannot turn successful completion into a false timeout.
            if _drain_reader_threads(
                process,
                readers,
                platform_tools,
                cleanup_deadline=absolute_deadline,
            ) and not (stdout_exceeded.is_set() or stderr_exceeded.is_set()):
                # A descendant may have escaped the worker process group while
                # retaining a pipe. Fail closed instead of returning partial
                # output as a successful worker completion.  Do not override
                # an existing output-limit-exceeded signal with timed_out.
                timed_out = True
            for reader, stream in zip(
                readers, (process.stdout, process.stderr), strict=True
            ):
                if not reader.is_alive():
                    stream.close()
                # A buffered reader can hold its stream lock inside read1 while
                # an escaped descendant retains the write end. In that case
                # the daemon reader owns eventual close; the coordinator must
                # not block past the request deadline trying to acquire the
                # same lock.
                reader.join(timeout=max(0.0, absolute_deadline - time.monotonic()))

    return BoundedProcessResult(
        returncode=process.returncode,
        stdout=bytes(stdout),
        stderr=bytes(stderr),
        stdout_exceeded=stdout_exceeded.is_set(),
        stderr_exceeded=stderr_exceeded.is_set(),
        timed_out=timed_out,
        cancelled=cancelled,
    )


def run_bounded_worker_dialogue[ValueT](
    command: Sequence[str],
    dialogue: Callable[[BoundedWorkerDialogue], ValueT],
    *,
    absolute_deadline: float,
    environment: Mapping[str, str],
    stdout_limit: int,
    stderr_limit: int,
    cwd: str | None = None,
) -> BoundedWorkerDialogueCompleted[ValueT]:
    """Run one adaptive child exchange inside an already supervised worker.

    The nested child deliberately inherits the worker's process group. Local
    failures kill and reap only that immediate child; the outer
    :func:`run_bounded_process` supervisor remains responsible for the whole
    group. The callback receives no process object, PID, or reusable session.

    ``absolute_deadline`` is a finite :func:`time.monotonic` deadline shared by
    launch admission, every read and write, child exit, stream drain, and
    cleanup. ``stdout_limit`` is cumulative across all frames; ``frame_limit``
    on :meth:`BoundedWorkerDialogue.read_until` bounds one unread frame.
    Python commands use the same package import binding as
    :func:`run_bounded_process`.
    """

    if stdout_limit < 0 or stderr_limit < 0:
        raise ValueError("worker dialogue output limits must be nonnegative")
    if not math.isfinite(absolute_deadline):
        raise ValueError("worker dialogue deadline must be finite")

    started = time.monotonic()
    if started >= absolute_deadline:
        raise BoundedWorkerDialogueError(
            BoundedWorkerDialogueErrorReason.DEADLINE_EXPIRED,
            stderr=b"",
        )
    cleanup_allowance = min(
        _PIPE_DRAIN_GRACE_SECONDS,
        (absolute_deadline - started) / 100,
    )
    execution_deadline = absolute_deadline - cleanup_allowance
    if time.monotonic() >= execution_deadline:
        raise BoundedWorkerDialogueError(
            BoundedWorkerDialogueErrorReason.DEADLINE_EXPIRED,
            stderr=b"",
        )

    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=_command_environment(command, environment),
            cwd=cwd,
            start_new_session=False,
            creationflags=0,
        )
    except OSError as exc:
        raise BoundedWorkerDialogueError(
            BoundedWorkerDialogueErrorReason.START_FAILED,
            stderr=b"",
        ) from exc

    state = _BoundedWorkerDialogueState(
        process,
        execution_deadline=execution_deadline,
        absolute_deadline=absolute_deadline,
        stdout_limit=stdout_limit,
        stderr_limit=stderr_limit,
    )
    bounded_dialogue = BoundedWorkerDialogue(state)
    try:
        state.start()
        if time.monotonic() >= execution_deadline:
            raise _WorkerDialogueControlError(
                BoundedWorkerDialogueErrorReason.DEADLINE_EXPIRED
            )
        value = dialogue(bounded_dialogue)
        state.finish()
    except _WorkerDialogueControlError as exc:
        state.deactivate()
        state.cleanup(kill=True)
        raise BoundedWorkerDialogueError(exc.reason, stderr=state.stderr) from None
    except BaseException:
        state.deactivate()
        state.cleanup(kill=True)
        raise

    stdout_bytes = state.stdout_bytes
    state.deactivate()
    state.cleanup(kill=False)
    return BoundedWorkerDialogueCompleted(value=value, stdout_bytes=stdout_bytes)


def _command_environment(
    command: Sequence[str], environment: Mapping[str, str]
) -> Mapping[str, str]:
    # Bind only the host interpreter, before any prlimit wrapping. Do not infer
    # Python from executable names or resolved symlinks: another virtualenv may
    # share the binary while owning a different package environment.
    if command and command[0] == sys.executable and "PYTHONPATH" not in environment:
        return {
            **environment,
            "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
        }
    return environment


def worker_environment(
    *,
    extra_variables: tuple[str, ...] = (),
    overrides: dict[str, str] | None = None,
    path_prefix: str | None = None,
    locale: str = _DEFAULT_LOCALE,
) -> dict[str, str]:
    """Build a deterministic subprocess environment without ambient leakage."""

    environment: dict[str, str] = {
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TZ": "UTC",
        "LANG": locale,
        "LC_ALL": locale,
    }
    for name in extra_variables:
        value = os.environ.get(name)
        if value is not None:
            environment[name] = value
    if path_prefix:
        environment["PATH"] = path_prefix
    if overrides:
        environment.update(overrides)
    return environment
