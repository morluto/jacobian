"""Process-contract tests for the isolated polynomial-ideal SymPy worker."""

from __future__ import annotations

import json
from typing import Any

import pytest

from jacobian.math.polynomials.ideals import _sympy_process
from jacobian.process import BoundedProcessResult


def _completed(
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    returncode: int | None = 0,
    stdout_exceeded: bool = False,
    stderr_exceeded: bool = False,
    timed_out: bool = False,
    cancelled: bool = False,
) -> BoundedProcessResult:
    return BoundedProcessResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_exceeded=stdout_exceeded,
        stderr_exceeded=stderr_exceeded,
        timed_out=timed_out,
        cancelled=cancelled,
    )


def _success_for_request(input_bytes: bytes) -> bytes:
    request = json.loads(input_bytes)
    return json.dumps(
        {
            "protocol_version": request["protocol_version"],
            "request_digest": request["request_digest"],
            "status": "ok",
            "value": 1,
        },
        separators=(",", ":"),
    ).encode("ascii")


def test_success_response_is_bound_to_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(*args: Any, **kwargs: Any) -> BoundedProcessResult:
        return _completed(stdout=_success_for_request(kwargs["input_bytes"]))

    monkeypatch.setattr(_sympy_process, "run_bounded_process", run)
    assert _sympy_process._run_sympy_kernel({"mode": "probe"}, 1)["value"] == 1


@pytest.mark.parametrize(
    ("result", "error", "message"),
    [
        (_completed(timed_out=True), _sympy_process._SympyKernelTimeoutError, None),
        (_completed(cancelled=True), _sympy_process._SympyKernelCancelledError, None),
        (
            _completed(stdout_exceeded=True),
            _sympy_process._ResultLimitExceededError,
            "channel bound",
        ),
        (
            _completed(stderr_exceeded=True),
            _sympy_process._SympyKernelError,
            "diagnostic channel",
        ),
        (
            _completed(returncode=2, stdout=b"{}", stderr=b"crashed"),
            _sympy_process._SympyKernelError,
            "crashed",
        ),
        (
            _completed(stdout=b"[]"),
            _sympy_process._SympyKernelError,
            "must be an object",
        ),
    ],
)
def test_process_failures_are_classified_before_result_use(
    monkeypatch: pytest.MonkeyPatch,
    result: BoundedProcessResult,
    error: type[Exception],
    message: str | None,
) -> None:
    monkeypatch.setattr(
        _sympy_process,
        "run_bounded_process",
        lambda *args, **kwargs: result,
    )
    with pytest.raises(error, match=message):
        _sympy_process._run_sympy_kernel({"mode": "probe"}, 1)


def test_response_digest_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = json.dumps(
        {
            "protocol_version": 1,
            "request_digest": "0" * 64,
            "status": "ok",
        }
    ).encode("ascii")
    monkeypatch.setattr(
        _sympy_process,
        "run_bounded_process",
        lambda *args, **kwargs: _completed(stdout=response),
    )
    with pytest.raises(_sympy_process._SympyKernelError, match="not bound"):
        _sympy_process._run_sympy_kernel({"mode": "probe"}, 1)
