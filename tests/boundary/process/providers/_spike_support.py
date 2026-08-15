from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence

from benchmarks.tooling.command_runner import ToolCommandResult, ToolCommandStatus


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _result(
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    returncode: int | None = 0,
    timed_out: bool = False,
    cancelled: bool = False,
    stdout_exceeded: bool = False,
) -> ToolCommandResult:
    if cancelled:
        status = ToolCommandStatus.CANCELLED
    elif timed_out:
        status = ToolCommandStatus.TIMED_OUT
    elif stdout_exceeded:
        status = ToolCommandStatus.OUTPUT_LIMIT_EXCEEDED
    else:
        status = ToolCommandStatus.EXITED
    return ToolCommandResult(
        status=status,
        exit_code=returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_exceeded=stdout_exceeded,
    )


def _runner(
    outcomes: Sequence[ToolCommandResult],
) -> Callable[..., ToolCommandResult]:
    remaining = iter(outcomes)

    def run(*_args: object, **_kwargs: object) -> ToolCommandResult:
        return next(remaining)

    return run
