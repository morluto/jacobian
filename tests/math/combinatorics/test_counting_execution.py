"""Counting control outcomes use the public operation execution errors."""

from threading import Event

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_cancellation,
    request_execution,
)
from jacobian.math.combinatorics import _counting_process
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
        lambda *a, **kw: BoundedProcessResult(
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


def test_expired_counting_request_before_startup() -> None:
    with request_execution(0), pytest.raises(OperationExecutionTimeoutError):
        _counting_process.evaluate_count("comb", 4, 2)


def test_cancelled_counting_request_before_startup() -> None:
    event = Event()
    event.set()
    with request_cancellation(event), pytest.raises(OperationExecutionCancelledError):
        _counting_process.evaluate_count("comb", 4, 2)
