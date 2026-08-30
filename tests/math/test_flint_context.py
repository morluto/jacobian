from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import monotonic

import pytest
from flint import arb, ctx

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian._flint import flint_workprec
from jacobian.process import bounded_process_cancellation


def _exp_one_interval() -> tuple[tuple[int, int], tuple[int, int]]:
    value = arb(1).exp()
    lower_mantissa, lower_exponent = value.lower().man_exp()
    upper_mantissa, upper_exponent = value.upper().man_exp()
    return (
        (int(lower_mantissa), int(lower_exponent)),
        (int(upper_mantissa), int(upper_exponent)),
    )


def test_flint_workprec_serializes_real_arb_precision_and_restores_context() -> None:
    original_precision = ctx.prec
    with flint_workprec(32):
        expected_low_precision = _exp_one_interval()
    with flint_workprec(512):
        expected_high_precision = _exp_one_interval()

    first_entered = Event()
    second_attempting = Event()
    second_entered = Event()

    def low_precision_worker() -> tuple[tuple[int, int], tuple[int, int]]:
        with flint_workprec(32):
            first_entered.set()
            assert second_attempting.wait(timeout=1)
            assert not second_entered.wait(timeout=0.25)
            assert ctx.prec == 32
            return _exp_one_interval()

    def high_precision_worker() -> tuple[tuple[int, int], tuple[int, int]]:
        assert first_entered.wait(timeout=1)
        second_attempting.set()
        with flint_workprec(512):
            second_entered.set()
            assert ctx.prec == 512
            return _exp_one_interval()

    with ThreadPoolExecutor(max_workers=2) as workers:
        low_future = workers.submit(low_precision_worker)
        high_future = workers.submit(high_precision_worker)
        low_precision = low_future.result(timeout=2)
        high_precision = high_future.result(timeout=2)

    assert low_precision == expected_low_precision
    assert high_precision == expected_high_precision
    assert ctx.prec == original_precision


def test_flint_workprec_is_reentrant_and_restores_nested_precision() -> None:
    original_precision = ctx.prec

    with flint_workprec(64):
        outer_interval = _exp_one_interval()
        assert ctx.prec == 64
        with flint_workprec(256, deadline=monotonic() + 0.25):
            inner_interval = _exp_one_interval()
            assert ctx.prec == 256
        assert ctx.prec == 64

    assert outer_interval != inner_interval
    assert ctx.prec == original_precision


def test_flint_workprec_rejects_an_expired_native_deadline() -> None:
    original_precision = ctx.prec

    with flint_workprec(64):
        with (
            pytest.raises(
                OperationExecutionTimeoutError,
                match="python-flint precision context",
            ),
            flint_workprec(256, deadline=monotonic() - 1.0),
        ):
            raise AssertionError("expired native deadline entered the context")
        assert ctx.prec == 64

    assert ctx.prec == original_precision


def test_flint_workprec_restores_precision_after_body_failure() -> None:
    original_precision = ctx.prec

    with pytest.raises(RuntimeError, match="test body failed"), flint_workprec(80):
        assert ctx.prec == 80
        _exp_one_interval()
        raise RuntimeError("test body failed")

    assert ctx.prec == original_precision


def test_flint_workprec_lock_wait_uses_the_request_deadline() -> None:
    original_precision = ctx.prec
    holder_entered = Event()
    release_holder = Event()

    def hold_context() -> tuple[tuple[int, int], tuple[int, int]]:
        with flint_workprec(96):
            interval = _exp_one_interval()
            holder_entered.set()
            release_holder.wait(timeout=0.4)
            return interval

    with ThreadPoolExecutor(max_workers=1) as workers:
        holder = workers.submit(hold_context)
        assert holder_entered.wait(timeout=1)
        wait_started = monotonic()
        try:
            with request_execution(wait_started):
                bind_request_deadline(wait_started + 0.05)
                with (
                    pytest.raises(
                        OperationExecutionTimeoutError,
                        match="python-flint precision context",
                    ),
                    flint_workprec(128),
                ):
                    raise AssertionError("expired lock wait entered the context")
        finally:
            release_holder.set()
        holder.result(timeout=1)

    assert ctx.prec == original_precision


def test_flint_workprec_without_deadline_waits_for_the_context() -> None:
    holder_entered = Event()
    release_holder = Event()

    def hold_context() -> None:
        with flint_workprec(96):
            holder_entered.set()
            assert release_holder.wait(timeout=1)

    def wait_for_context() -> int:
        assert holder_entered.wait(timeout=1)
        with flint_workprec(128):
            return ctx.prec

    with ThreadPoolExecutor(max_workers=2) as workers:
        holder = workers.submit(hold_context)
        waiter = workers.submit(wait_for_context)
        assert holder_entered.wait(timeout=1)
        assert not waiter.done()
        release_holder.set()
        holder.result(timeout=1)
        assert waiter.result(timeout=1) == 128


def test_flint_workprec_queued_wait_observes_request_cancellation() -> None:
    original_precision = ctx.prec
    holder_entered = Event()
    release_holder = Event()
    waiter_attempting = Event()
    waiter_entered = Event()
    cancellation = Event()

    def hold_context() -> tuple[tuple[int, int], tuple[int, int]]:
        with flint_workprec(96):
            interval = _exp_one_interval()
            holder_entered.set()
            assert release_holder.wait(timeout=2)
            return interval

    def wait_for_context() -> tuple[tuple[int, int], tuple[int, int]]:
        assert holder_entered.wait(timeout=1)
        with bounded_process_cancellation(cancellation):
            waiter_attempting.set()
            with flint_workprec(256, deadline=monotonic() + 2.0):
                waiter_entered.set()
                return _exp_one_interval()

    with ThreadPoolExecutor(max_workers=2) as workers:
        holder = workers.submit(hold_context)
        assert holder_entered.wait(timeout=1)
        waiter = workers.submit(wait_for_context)
        assert waiter_attempting.wait(timeout=1)
        assert not waiter.done()
        cancellation.set()
        try:
            with pytest.raises(
                OperationExecutionCancelledError,
                match="python-flint precision context",
            ):
                waiter.result(timeout=1)
            assert not waiter_entered.is_set()
        finally:
            release_holder.set()
        holder.result(timeout=1)
        assert not waiter_entered.is_set()

    assert ctx.prec == original_precision


def test_flint_workprec_rechecks_cancellation_after_lock_acquisition() -> None:
    class GatedCancellation:
        def __init__(self) -> None:
            self._cancelled = Event()
            self._allow_first_return = Event()
            self.first_check_read = Event()
            self._is_first_check = True

        def is_set(self) -> bool:
            observed = self._cancelled.is_set()
            if self._is_first_check:
                self._is_first_check = False
                self.first_check_read.set()
                assert self._allow_first_return.wait(timeout=1)
            return observed

        def set(self) -> None:
            self._cancelled.set()

        def allow_first_return(self) -> None:
            self._allow_first_return.set()

    original_precision = ctx.prec
    holder_entered = Event()
    release_holder = Event()
    waiter_entered = Event()
    cancellation = GatedCancellation()

    def hold_context() -> tuple[tuple[int, int], tuple[int, int]]:
        with flint_workprec(96):
            interval = _exp_one_interval()
            holder_entered.set()
            assert release_holder.wait(timeout=2)
            return interval

    def wait_for_context() -> tuple[tuple[int, int], tuple[int, int]]:
        with (
            bounded_process_cancellation(cancellation),
            flint_workprec(256, deadline=monotonic() + 2.0),
        ):
            waiter_entered.set()
            return _exp_one_interval()

    with ThreadPoolExecutor(max_workers=2) as workers:
        holder = workers.submit(hold_context)
        assert holder_entered.wait(timeout=1)
        waiter = workers.submit(wait_for_context)
        try:
            assert cancellation.first_check_read.wait(timeout=1)
            cancellation.set()
            release_holder.set()
            cancellation.allow_first_return()
            with pytest.raises(
                OperationExecutionCancelledError,
                match="python-flint precision context",
            ):
                waiter.result(timeout=1)
        finally:
            cancellation.allow_first_return()
            release_holder.set()
        holder.result(timeout=1)
        assert not waiter_entered.is_set()

    assert ctx.prec == original_precision
