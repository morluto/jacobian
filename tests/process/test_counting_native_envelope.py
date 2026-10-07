"""Native exact counts use one allowance across admission and execution."""

from __future__ import annotations

import time
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from jacobian import _execution
from jacobian._execution import OperationExecutionTimeoutError, request_execution
from jacobian.math import combinatorics
from jacobian.math.combinatorics import _counting_process, operations


@pytest.mark.parametrize(
    ("name", "estimator"),
    [
        ("binomial", "_binomial_coefficient_digit_bound"),
        ("permutations", "_falling_factorial_digit_bound"),
        ("compositions", "_binomial_coefficient_digit_bound"),
    ],
)
@pytest.mark.parametrize(
    "caller_scope", [False, True], ids=["native", "caller-control"]
)
def test_native_counting_admission_cannot_receive_a_fresh_worker_clock(
    monkeypatch: pytest.MonkeyPatch, name: str, estimator: str, caller_scope: bool
) -> None:
    started = time.monotonic()
    clock = [started]
    computed_bounds: list[int] = []
    estimate = getattr(operations, estimator)

    def complete_admission(*args: Any, **kwargs: Any) -> int:
        bound = estimate(*args, **kwargs)
        assert type(bound) is int
        computed_bounds.append(bound)
        # The actual mathematical estimate is retained. Only its elapsed
        # allowance is controlled, independently of host scheduling speed.
        clock[0] = started + _counting_process._COUNTING_WALL_SECONDS + 1
        return bound

    monkeypatch.setattr(operations, estimator, complete_admission)
    local_time = SimpleNamespace(monotonic=lambda: clock[0])
    for owner in (operations, _counting_process, _execution):
        monkeypatch.setattr(owner, "time", local_time)

    with (
        request_execution(started) if caller_scope else nullcontext(),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        getattr(combinatorics, name)(4, 2)

    assert len(computed_bounds) == 1
    assert computed_bounds[0] > 0
    assert _execution.current_request_execution() is None


@pytest.mark.parametrize(
    ("name", "expected"), [("binomial", 6), ("permutations", 12), ("compositions", 3)]
)
def test_native_counting_real_worker_controls(name: str, expected: int) -> None:
    assert getattr(combinatorics, name)(4, 2) == expected
    assert _execution.current_request_execution() is None


@pytest.mark.parametrize("name", ["binomial", "permutations", "compositions"])
def test_native_counting_result_construction_respects_the_same_deadline(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    started = time.monotonic()
    deadline = started + 30
    clock = [started]
    converted: list[int] = []
    run = _counting_process.evaluate_count

    def convert(value: str) -> int:
        parsed = int(value)
        converted.append(parsed)
        clock[0] = deadline + 1
        return parsed

    def finish_worker(*args: Any, **kwargs: Any) -> str:
        result = run(*args, **kwargs)
        # Change only the existing native integer conversion after the real
        # worker has delivered its own count; retain Python's actual integer.
        monkeypatch.setattr(operations, "int", convert, raising=False)
        return result

    monkeypatch.setattr(operations, "evaluate_count", finish_worker)
    monkeypatch.setattr(_execution, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    with (
        request_execution(started, outer_deadline=deadline),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        getattr(combinatorics, name)(4, 2)
    assert len(converted) == 1
    assert converted[0] > 0
    assert _execution.current_request_execution() is None
