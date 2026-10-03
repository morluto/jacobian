"""Checkpoint provenance follows the limiting deadline, not observation time."""

from threading import Event
from types import SimpleNamespace

import pytest

from jacobian import _execution
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionStage,
    OperationExecutionTimeoutError,
    TimeoutOwner,
    bind_request_deadline,
    request_checkpoint,
    request_execution,
    request_stage,
)
from jacobian._worker_errors import bind_worker_deadline


@pytest.mark.parametrize(
    ("operation_deadline", "outer_deadline", "observed_at", "expected_owner"),
    [
        (105.0, 110.0, 107.0, TimeoutOwner.OPERATION_WALL),
        (105.0, 110.0, 111.0, TimeoutOwner.OPERATION_WALL),
        (110.0, 105.0, 107.0, TimeoutOwner.CALLER_DEADLINE),
        (110.0, 105.0, 111.0, TimeoutOwner.CALLER_DEADLINE),
        (105.0, None, 111.0, TimeoutOwner.OPERATION_WALL),
        (None, 105.0, 111.0, TimeoutOwner.CALLER_DEADLINE),
        (105.0, 105.0, 111.0, TimeoutOwner.CALLER_DEADLINE),
    ],
)
@pytest.mark.parametrize("stage", list(OperationExecutionStage))
def test_checkpoint_reports_the_limiting_deadline_owner(
    monkeypatch: pytest.MonkeyPatch,
    operation_deadline: float | None,
    outer_deadline: float | None,
    observed_at: float,
    expected_owner: TimeoutOwner,
    stage: OperationExecutionStage,
) -> None:
    monkeypatch.setattr(
        _execution, "time", SimpleNamespace(monotonic=lambda: observed_at)
    )
    with request_execution(100.0, outer_deadline=outer_deadline), request_stage(stage):
        if operation_deadline is not None:
            bind_request_deadline(operation_deadline)
        with pytest.raises(OperationExecutionTimeoutError) as caught:
            request_checkpoint("during a bounded phase")

    assert caught.value.timeout_owner is expected_owner
    assert caught.value.stage is stage
    assert caught.value.elapsed_seconds == observed_at - 100.0
    assert caught.value.configured_seconds is None


def test_checkpoint_preserves_an_explicit_outer_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_execution, "time", SimpleNamespace(monotonic=lambda: 111.0))
    with request_execution(
        100.0, outer_deadline=105.0, timeout_owner=TimeoutOwner.BACKEND_TIMEOUT
    ):
        bind_request_deadline(110.0)
        with pytest.raises(OperationExecutionTimeoutError) as caught:
            request_checkpoint("during backend execution")

    assert caught.value.timeout_owner is TimeoutOwner.BACKEND_TIMEOUT


def test_checkpoint_accepts_an_unexpired_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_execution, "time", SimpleNamespace(monotonic=lambda: 104.0))
    with request_execution(100.0, outer_deadline=110.0) as execution:
        bind_request_deadline(105.0)
        request_checkpoint("before the limiting deadline")
        assert execution.deadline == 105.0


def test_checkpoint_cancellation_wins_after_both_deadlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_execution, "time", SimpleNamespace(monotonic=lambda: 111.0))
    cancellation = Event()
    cancellation.set()
    with request_execution(
        100.0, outer_deadline=110.0, cancellation_signal=cancellation
    ):
        bind_request_deadline(105.0)
        with pytest.raises(OperationExecutionCancelledError) as caught:
            request_checkpoint(
                "after result projection",
                public_stage=OperationExecutionStage.RESULT_PROJECTION,
            )

    assert caught.value.stage is OperationExecutionStage.RESULT_PROJECTION


def test_inherited_worker_deadline_does_not_invent_a_configured_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [108.0]
    monkeypatch.setattr(_execution, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    # The worker starts later than its parent and inherits only an absolute
    # deadline. Its remaining time is not the configured full request budget.
    with request_execution(108.0):
        bind_worker_deadline({"_deadline": 110.0})
        clock[0] = 111.0
        with pytest.raises(OperationExecutionTimeoutError) as caught:
            request_checkpoint("during worker execution")

    assert caught.value.timeout_owner is TimeoutOwner.OPERATION_WALL
    assert caught.value.elapsed_seconds == 3.0
    assert caught.value.configured_seconds is None
