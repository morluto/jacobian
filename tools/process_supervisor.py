"""Supervise one child process tree and kill every descendant on timeout."""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

_GRACEFUL_SHUTDOWN_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class ProcessTreeResult:
    """Exit status of one supervised process tree."""

    exit_code: int
    timed_out: bool


def kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate a child and every descendant in its process group or job."""

    if os.name == "posix":
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=_GRACEFUL_SHUTDOWN_SECONDS)
        # The root process can exit after SIGTERM while a descendant that
        # ignored the signal remains. Always SIGKILL the group.
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        with suppress(subprocess.TimeoutExpired, ProcessLookupError):
            process.wait(timeout=_GRACEFUL_SHUTDOWN_SECONDS)
        return
    process.terminate()
    try:
        process.wait(timeout=_GRACEFUL_SHUTDOWN_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=_GRACEFUL_SHUTDOWN_SECONDS)


def run_process_tree(
    command: Sequence[str],
    *,
    timeout: float,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> ProcessTreeResult:
    """Run ``command`` in a new process group and kill the tree on timeout.

    Standard streams are inherited. The returned exit code is the child's
    actual status, or ``1`` when the outer deadline fires.
    """

    if timeout <= 0:
        raise ValueError("timeout must be positive")
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        start_new_session=os.name == "posix",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_tree(process)
        return ProcessTreeResult(exit_code=1, timed_out=True)
    return ProcessTreeResult(exit_code=int(returncode), timed_out=False)
