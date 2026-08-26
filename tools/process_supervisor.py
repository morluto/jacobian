"""Compatibility projection for bounded repository command execution."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tools.command_runner import (
    ToolCommandRequest,
    ToolCommandStatus,
    run_tool_command,
)

_PROCESS_OUTPUT_LIMIT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ProcessTreeResult:
    """Exit status of one repository command tree."""

    exit_code: int
    timed_out: bool


def _stream(stream: object) -> Callable[[bytes], None] | None:
    """Return a byte sink for an inherited text or binary stream."""

    binary = getattr(stream, "buffer", None)
    target = binary if binary is not None else stream
    write = getattr(target, "write", None)
    flush = getattr(target, "flush", None)
    if not callable(write):
        return None

    def sink(block: bytes) -> None:
        write(block if binary is not None else block.decode("utf-8", "replace"))
        if callable(flush):
            flush()

    return sink


def run_process_tree(
    command: Sequence[str],
    *,
    timeout: float,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> ProcessTreeResult:
    """Run one absolute repository command under the shared bounded runner."""

    if not command:
        raise ValueError("command must not be empty")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    request = ToolCommandRequest(
        executable=command[0],
        arguments=tuple(command[1:]),
        environment=dict(os.environ) if env is None else env,
        cwd=str(cwd.resolve()),
        timeout_seconds=timeout,
        stdout_limit_bytes=_PROCESS_OUTPUT_LIMIT_BYTES,
        stderr_limit_bytes=_PROCESS_OUTPUT_LIMIT_BYTES,
        stdout_sink=_stream(sys.stdout),
        stderr_sink=_stream(sys.stderr),
    )
    completed = run_tool_command(request)
    return ProcessTreeResult(
        exit_code=(
            completed.exit_code
            if completed.status is ToolCommandStatus.EXITED
            and completed.exit_code is not None
            else 1
        ),
        timed_out=completed.status is ToolCommandStatus.TIMED_OUT,
    )
