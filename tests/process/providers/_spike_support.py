from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.command_runner import (
    ToolCommandRequest,
    ToolCommandResult,
    ToolCommandStatus,
)


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


@dataclass(slots=True)
class _ExpectedRunner:
    outcomes: tuple[ToolCommandResult, ...]
    requests: list[ToolCommandRequest]
    expected_requests: tuple[ToolCommandRequest, ...] = ()
    on_call: Callable[[int, ToolCommandRequest], None] | None = None
    _index: int = 0

    def __call__(self, request: ToolCommandRequest) -> ToolCommandResult:
        if not isinstance(request, ToolCommandRequest):
            raise AssertionError("provider spike must submit one ToolCommandRequest")
        if self._index >= len(self.outcomes):
            raise AssertionError(f"unexpected provider command: {request!r}")
        if not self.expected_requests:
            raise AssertionError("strict runner expectations were not configured")
        expected = self.expected_requests[self._index]
        if request != expected:
            raise AssertionError(
                f"provider command {self._index + 1} mismatch:\n"
                f"expected {expected!r}\nactual   {request!r}"
            )
        self.requests.append(request)
        outcome = self.outcomes[self._index]
        self._index += 1
        if self.on_call is not None:
            self.on_call(self._index, request)
        return outcome

    def assert_finished(self) -> None:
        remaining = len(self.outcomes) - self._index
        if remaining:
            raise AssertionError(
                f"{remaining} expected provider command(s) were not run"
            )

    def expect(self, requests: Sequence[ToolCommandRequest]) -> None:
        expected = tuple(requests)
        if len(expected) != len(self.outcomes):
            raise AssertionError("request and outcome expectation counts must match")
        self.expected_requests = expected


def _runner(outcomes: Sequence[ToolCommandResult]) -> _ExpectedRunner:
    return _ExpectedRunner(tuple(outcomes), [])


def _request(
    executable: Path,
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: Path,
    timeout_seconds: float,
    stdin_bytes: bytes = b"",
    stdout_limit_bytes: int,
    stderr_limit_bytes: int,
) -> ToolCommandRequest:
    return ToolCommandRequest(
        executable=str(executable.resolve(strict=True)),
        arguments=tuple(arguments),
        environment=environment,
        cwd=str(cwd.resolve()),
        timeout_seconds=timeout_seconds,
        stdin_bytes=stdin_bytes,
        stdout_limit_bytes=stdout_limit_bytes,
        stderr_limit_bytes=stderr_limit_bytes,
    )


def _run_expected(
    run: Callable[[], dict[str, Any]],
    runner: _ExpectedRunner,
    requests: Sequence[ToolCommandRequest],
) -> dict[str, Any]:
    runner.expect(tuple(requests)[: len(runner.outcomes)])
    result = run()
    runner.assert_finished()
    return result
