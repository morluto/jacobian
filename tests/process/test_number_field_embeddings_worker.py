"""Failure semantics for the one-shot number-field embedding worker."""

from __future__ import annotations

import time

import pytest

import jacobian.process as process
from jacobian._execution import OperationExecutionTimeoutError, request_execution
from jacobian.math.number_theory.number_fields import (
    SimpleNumberFieldPresentation,
    embeddings,
)
from jacobian.process import BoundedProcessResult


def _gaussian_field() -> SimpleNumberFieldPresentation:
    return SimpleNumberFieldPresentation(coefficients_descending=("1", "0", "1"))


def test_embedding_worker_timeout_is_an_operational_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process,
        "run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        ),
    )

    with pytest.raises(OperationExecutionTimeoutError, match="during"):
        embeddings(_gaussian_field())


def test_deadline_expiry_immediately_before_worker_launch_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((119.0, 119.5, 120.5))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))

    with (
        request_execution(started_at=0.0),
        pytest.raises(OperationExecutionTimeoutError, match=r"before.*launch"),
    ):
        embeddings(_gaussian_field())


def test_embedding_worker_start_failure_does_not_escape_as_os_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> BoundedProcessResult:
        raise OSError("worker unavailable")

    monkeypatch.setattr(process, "run_bounded_process", unavailable)

    with pytest.raises(RuntimeError, match="could not be started"):
        embeddings(_gaussian_field())


def test_embedding_worker_rejects_malformed_protocol_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process,
        "run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            returncode=0,
            stdout=b'\x7b"kind":"complete"\x7d',
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        ),
    )

    with pytest.raises(RuntimeError, match="malformed output"):
        embeddings(_gaussian_field())
