"""Backend phase leases retain reserves inside the effective request deadline."""

from __future__ import annotations

import time
from contextlib import nullcontext
from threading import Event

import pytest

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionStage,
    OperationExecutionTimeoutError,
    TimeoutOwner,
    bind_request_deadline,
    lease_operation_phases,
    request_cancellation,
    request_execution,
    request_stage,
)


@pytest.mark.parametrize(
    ("outer_deadline", "bound_deadline", "wall_seconds", "expected_deadline"),
    [
        (None, None, 10.0, 110.0),
        (105.0, None, 10.0, 105.0),
        (None, 102.0, 10.0, 102.0),
        (108.0, 102.0, 10.0, 102.0),
        (102.0, 108.0, 10.0, 102.0),
        (105.0, 108.0, 2.0, 102.0),
    ],
)
def test_phase_reserves_fit_inside_the_effective_deadline(
    monkeypatch: pytest.MonkeyPatch,
    outer_deadline: float | None,
    bound_deadline: float | None,
    wall_seconds: float,
    expected_deadline: float,
) -> None:
    monkeypatch.setattr(time, "monotonic", lambda: 101.0)
    with request_execution(100.0, outer_deadline=outer_deadline) as execution:
        if bound_deadline is not None:
            bind_request_deadline(bound_deadline)
        enclosing_deadline = execution.deadline
        lease = lease_operation_phases(
            wall_seconds, admitted_response_bytes=400_000, validation_work=300_000
        )
        assert lease.operation_deadline == expected_deadline
        assert lease.delivery_reserve_seconds == pytest.approx(0.04)
        assert lease.teardown_reserve_seconds == pytest.approx(0.05)
        assert lease.backend_deadline == pytest.approx(expected_deadline - 0.09)
        assert execution.deadline == (
            enclosing_deadline if bound_deadline is not None else expected_deadline
        )


def test_repeated_leases_keep_the_enclosing_allowance_and_original_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    with request_execution(100.0) as execution:
        first = lease_operation_phases(
            10.0, admitted_response_bytes=0, validation_work=0
        )
        now = 101.0
        local = lease_operation_phases(
            2.0, admitted_response_bytes=0, validation_work=0
        )
        assert local.operation_deadline == 102.0
        assert execution.deadline == 110.0
        now = 102.0
        later = lease_operation_phases(
            30.0, admitted_response_bytes=0, validation_work=0
        )
        assert later == first
        assert execution.deadline == 110.0


def test_phase_lease_without_an_envelope_uses_the_current_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "monotonic", lambda: 101.0)
    lease = lease_operation_phases(10.0, admitted_response_bytes=0, validation_work=0)
    assert lease.operation_deadline == 111.0
    assert lease.backend_deadline == pytest.approx(110.96)


def test_exhausted_enclosing_reserves_raise_an_operational_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "monotonic", lambda: 101.0)
    with request_execution(100.0) as execution:
        bind_request_deadline(101.08)
        with pytest.raises(OperationExecutionTimeoutError) as error:
            lease_operation_phases(
                10.0, admitted_response_bytes=400_000, validation_work=300_000
            )
        assert execution.deadline == 101.08
    assert error.value.timeout_owner is TimeoutOwner.OPERATION_WALL
    assert error.value.configured_seconds == 10.0
    assert error.value.elapsed_seconds == 1.0


@pytest.mark.parametrize("envelope_signal", [False, True])
def test_cancellation_precedes_exhausted_reserves_in_the_current_stage(
    monkeypatch: pytest.MonkeyPatch, envelope_signal: bool
) -> None:
    monkeypatch.setattr(time, "monotonic", lambda: 101.0)
    signal = Event()
    signal.set()
    with (
        request_execution(
            100.0, cancellation_signal=signal if envelope_signal else None
        ),
        nullcontext() if envelope_signal else request_cancellation(signal),
        request_stage(OperationExecutionStage.RESULT_PROJECTION),
    ):
        bind_request_deadline(101.02)
        with pytest.raises(OperationExecutionCancelledError) as error:
            lease_operation_phases(10.0, admitted_response_bytes=0, validation_work=0)
    assert error.value.stage is OperationExecutionStage.RESULT_PROJECTION
