"""Bounded one-shot subprocess capture for concrete external operations.

Children run in their own process group. Reader threads cap retained output and
terminate the whole group as soon as either stream exceeds its limit. Timeouts
also terminate descendants rather than only the immediate worker.

It is a small low-level primitive for genuinely isolated one-shot commands.
Each domain operation owns its command, temporary files, input, and conversion
to a typed mathematical result. There are no process sessions or protocols.
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
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

__all__ = [
    "BoundedProcessResult",
    "ProcessPlatformTools",
    "ProcessResourceLimits",
    "bounded_process_cancellation",
    "bounded_process_cancelled",
    "run_bounded_process",
]

_CANCELLATION_EVENT: ContextVar[threading.Event | None] = ContextVar(
    "jacobian_bounded_process_cancellation_event",
    default=None,
)
_PIPE_DRAIN_GRACE_SECONDS = 0.5
_RESOURCE_POLL_SECONDS = 0.1


@dataclass(frozen=True, slots=True)
class BoundedProcessResult:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    stdout_exceeded: bool
    stderr_exceeded: bool
    timed_out: bool
    cancelled: bool = False
    peak_rss_bytes: int | None = None


def _read_proc_status(pid_dir: Path) -> tuple[int, int] | None:
    """Return ``(parent_pid, rss_bytes)`` for one ``/proc/<pid>`` entry."""

    try:
        status = (pid_dir / "status").read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return None
    parent = 0
    rss_kb = 0
    for line in status.splitlines():
        if line.startswith("PPid:"):
            try:
                parent = int(line.split()[1])
            except (IndexError, ValueError):
                return None
        elif line.startswith("VmRSS:"):
            try:
                rss_kb = int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return parent, rss_kb * 1024


def _read_proc_pss_bytes(pid_dir: Path) -> int | None:
    """Return proportional resident memory from ``smaps_rollup`` when available.

    Summing ``VmRSS`` for a process tree counts shared Lean and Mathlib pages
    once for every process that maps them.  PSS apportions those shared pages,
    so it represents the tree's actual memory footprint much more closely.
    """

    try:
        lines = (pid_dir / "smaps_rollup").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    for line in lines:
        if not line.startswith("Pss:"):
            continue
        try:
            return int(line.split()[1]) * 1024
        except (IndexError, ValueError):
            return None
    return None


def _collect_linux_processes() -> dict[int, tuple[int, int]] | None:
    """Read all ``/proc`` process statuses, or ``None`` if procfs is unavailable."""

    processes: dict[int, tuple[int, int]] = {}
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            parsed = _read_proc_status(entry)
            if parsed is not None:
                processes[int(entry.name)] = parsed
    except OSError:
        return None
    return processes


def _descendant_pids(processes: dict[int, tuple[int, int]], root_pid: int) -> set[int]:
    """Collect ``root_pid`` plus all transitive children."""

    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _rss) in processes.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def _linux_process_tree_rss_bytes(root_pid: int) -> int | None:
    """Return current RSS for a Linux process tree, or ``None`` elsewhere."""

    if not sys.platform.startswith("linux"):
        return None
    processes = _collect_linux_processes()
    if processes is None:
        return None
    descendants = _descendant_pids(processes, root_pid)
    return sum(processes.get(pid, (0, 0))[1] for pid in descendants)


def _linux_process_tree_memory_bytes(root_pid: int) -> int | None:
    """Return proportional memory for a Linux process tree.

    ``smaps_rollup`` PSS avoids counting shared pages once for every process
    that maps them. Constrained or older procfs installations fall back to
    VmRSS, retaining a conservative bound.
    """

    if not sys.platform.startswith("linux"):
        return None
    processes = _collect_linux_processes()
    if processes is None:
        return None
    descendants = _descendant_pids(processes, root_pid)
    memory_bytes = 0
    for pid in descendants:
        pss_bytes = _read_proc_pss_bytes(Path("/proc") / str(pid))
        memory_bytes += (
            processes.get(pid, (0, 0))[1] if pss_bytes is None else pss_bytes
        )
    return memory_bytes


@contextmanager
def bounded_process_cancellation(
    event: threading.Event,
) -> Iterator[None]:
    """Bind cooperative subprocess cancellation to the current worker context."""

    token = _CANCELLATION_EVENT.set(event)
    try:
        yield
    finally:
        _CANCELLATION_EVENT.reset(token)


def bounded_process_cancelled() -> bool:
    """Report whether the current operation worker has lost its client."""

    event = _CANCELLATION_EVENT.get()
    return event is not None and event.is_set()


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
    cancellation_event: threading.Event | None,
    platform_tools: ProcessPlatformTools | None,
) -> tuple[bool, bool, int | None]:
    """Poll the child until exit, timeout, or cancellation.

    Returns ``(timed_out, cancelled, peak_rss_bytes)``.
    """

    timed_out = False
    cancelled = False
    peak_rss_bytes: int | None = None
    while process.poll() is None:
        current_rss = _linux_process_tree_rss_bytes(process.pid)
        if current_rss is not None:
            peak_rss_bytes = max(peak_rss_bytes or 0, current_rss)
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
    return timed_out, cancelled, peak_rss_bytes


def _drain_reader_threads(
    process: subprocess.Popen[bytes],
    readers: tuple[threading.Thread, ...],
    platform_tools: ProcessPlatformTools | None,
) -> bool:
    """Kill the tree, drain readers, and return ``True`` if any survived."""

    _kill_process_tree(process, platform_tools)
    drain_deadline = time.monotonic() + _PIPE_DRAIN_GRACE_SECONDS
    for reader in readers:
        reader.join(timeout=max(0.0, drain_deadline - time.monotonic()))
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
    cancellation_event: threading.Event | None = None,
) -> BoundedProcessResult:
    """Run a child with bounded output, time, lifetime, and supported resources.

    *cwd* is an optional absolute working directory passed to the child; when
    ``None`` the child inherits the engine's cwd.  *platform_tools* carries
    bootstrap-resolved absolute helper paths (prlimit, taskkill); the engine
    never discovers executables.  *cancellation_event* overrides the context-bound cancellation
    event for explicit callers; when ``None`` the engine uses the
    context-bound event set by :func:`bounded_process_cancellation`.
    """

    if stdout_limit < 0 or stderr_limit < 0:
        raise ValueError("subprocess output limits must be nonnegative")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("subprocess timeout must be positive")

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
        cancellation_event = _CANCELLATION_EVENT.get()

    prlimit_executable = (
        platform_tools.prlimit_executable if platform_tools is not None else None
    )

    with tempfile.TemporaryFile() as stdin_file:
        stdin_file.write(input_bytes)
        stdin_file.seek(0)
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
            env=environment,
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

        deadline = time.monotonic() + timeout_seconds
        try:
            timed_out, cancelled, peak_rss_bytes = _monitor_bounded_process(
                process,
                deadline=deadline,
                cancellation_event=cancellation_event,
                platform_tools=platform_tools,
            )
        finally:
            # A clean worker may leave descendants holding inherited pipe
            # handles. Terminate the process group before draining so those
            # handles cannot turn successful completion into a false timeout.
            if _drain_reader_threads(process, readers, platform_tools) and not (
                stdout_exceeded.is_set() or stderr_exceeded.is_set()
            ):
                # A descendant may have escaped the worker process group while
                # retaining a pipe. Fail closed instead of returning partial
                # output as a successful worker completion.  Do not override
                # an existing output-limit-exceeded signal with timed_out.
                timed_out = True
            process.stdout.close()
            process.stderr.close()
            for reader in readers:
                reader.join(timeout=0.1)

    return BoundedProcessResult(
        returncode=process.returncode,
        stdout=bytes(stdout),
        stderr=bytes(stderr),
        stdout_exceeded=stdout_exceeded.is_set(),
        stderr_exceeded=stderr_exceeded.is_set(),
        timed_out=timed_out,
        cancelled=cancelled,
        peak_rss_bytes=peak_rss_bytes,
    )


# ---------------------------------------------------------------------------
