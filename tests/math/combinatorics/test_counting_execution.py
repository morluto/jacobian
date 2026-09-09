"""Counting control outcomes use the public operation execution errors."""

import time
from threading import Event

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_cancellation,
    request_execution,
)
from jacobian.math.combinatorics import _counting_process
from jacobian.math.combinatorics.operations import canonical_binomial
from jacobian.process import BoundedProcessResult


@pytest.mark.parametrize(
    "cancelled,timed_out,error",
    [
        (True, False, OperationExecutionCancelledError),
        (False, True, OperationExecutionTimeoutError),
    ],
)
def test_counting_worker_control_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    cancelled: bool,
    timed_out: bool,
    error: type[Exception],
) -> None:
    monkeypatch.setattr(
        _counting_process,
        "run_bounded_process",
        lambda *a, **_kwargs: BoundedProcessResult(
            returncode=-1,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            cancelled=cancelled,
            timed_out=timed_out,
        ),
    )
    with pytest.raises(error) as raised:
        _counting_process.evaluate_count("comb", 4, 2)
    assert isinstance(
        raised.value, (OperationExecutionCancelledError, OperationExecutionTimeoutError)
    )
    assert raised.value.stage == "operation_execution"


@pytest.mark.parametrize("now", [-1000.0, 0.0, 60.0, 1_000_000.0])
def test_expired_counting_request_before_startup(
    monkeypatch: pytest.MonkeyPatch, now: float
) -> None:
    monkeypatch.setattr(time, "monotonic", lambda: now)
    started = now - _counting_process._COUNTING_WALL_SECONDS - 1
    with request_execution(started), pytest.raises(OperationExecutionTimeoutError):
        _counting_process.evaluate_count("comb", 4, 2)


def test_cancelled_counting_request_before_startup() -> None:
    event = Event()
    event.set()
    with request_cancellation(event), pytest.raises(OperationExecutionCancelledError):
        _counting_process.evaluate_count("comb", 4, 2)


def test_counting_preserves_a_shorter_caller_deadline() -> None:
    started = time.monotonic()
    caller_deadline = started + 10.0
    with request_execution(started):
        bind_request_deadline(caller_deadline)
        assert canonical_binomial(4, 2) == 6
        execution = current_request_execution()
        active_deadline = execution.deadline if execution is not None else None

    assert execution is not None
    assert active_deadline == caller_deadline


def test_expired_caller_deadline_is_not_replaced_by_counting_owner_limit() -> None:
    started = time.monotonic()
    with request_execution(started):
        bind_request_deadline(started - 1.0)
        with pytest.raises(OperationExecutionTimeoutError):
            canonical_binomial(4, 2)
