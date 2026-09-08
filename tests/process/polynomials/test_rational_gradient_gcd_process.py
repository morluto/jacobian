"""Process-boundary tests for killable denominator-derivative GCDs."""

from __future__ import annotations

from time import monotonic
from typing import Any

import pytest
from sympy import symbols

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.math.polynomials._conversions import rational_function_from_sympy
from jacobian.math.polynomials.rational_functions.gradient._gcd_process import (
    forced_denominator_derivative_gcds,
)
from jacobian.math.polynomials.rational_functions.gradient.operations import gradient
from jacobian.process import BoundedProcessResult, ProcessResourceLimits


def test_linear_power_gcd_runs_in_the_killable_worker() -> None:
    x = symbols("x")
    source = rational_function_from_sympy(1 / (x + 1) ** 33, ("x",))
    started = monotonic()
    with request_execution(started):
        bind_request_deadline(started + 60)
        factors = forced_denominator_derivative_gcds(source.denominator, 1)
    assert factors[0].degrees == (32,)
    assert factors[0].total_degree == 32
    result = gradient(source)
    assert result.partial_derivatives[0] == rational_function_from_sympy(
        -33 / (x + 1) ** 34, ("x",)
    )


def test_gcd_worker_timeout_uses_the_remaining_request_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian import process

    observed: dict[str, Any] = {}

    def timed_out(*args: Any, **kwargs: Any) -> BoundedProcessResult:
        observed.update(kwargs)
        return BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        )

    monkeypatch.setattr(process, "run_bounded_process", timed_out)
    x = symbols("x")
    source = rational_function_from_sympy(1 / (x + 1), ("x",))
    started = monotonic()
    deadline = started + 5.0
    with (
        request_execution(started),
        pytest.raises(
            OperationExecutionTimeoutError,
            match="during denominator-derivative gcd",
        ),
    ):
        bind_request_deadline(deadline)
        forced_denominator_derivative_gcds(source.denominator, 1)

    assert 0 < observed["timeout_seconds"] <= 5.0
    resource_limits = observed["resource_limits"]
    assert isinstance(resource_limits, ProcessResourceLimits)
    assert resource_limits.cpu_seconds is not None
    assert resource_limits.address_space_bytes is not None
    assert resource_limits.file_size_bytes is not None
    assert (
        str(observed["cwd"])
        .split("/")[-1]
        .startswith("jacobian-rational-gradient-gcd-")
    )


def test_gcd_worker_cancel_is_a_typed_non_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian import process

    def cancelled(*args: Any, **kwargs: Any) -> BoundedProcessResult:
        return BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
            cancelled=True,
        )

    monkeypatch.setattr(process, "run_bounded_process", cancelled)
    x = symbols("x")
    source = rational_function_from_sympy(1 / (x + 1), ("x",))
    started = monotonic()
    with (
        request_execution(started),
        pytest.raises(
            OperationExecutionCancelledError,
            match="cancelled during denominator-derivative gcd",
        ),
    ):
        bind_request_deadline(started + 60)
        forced_denominator_derivative_gcds(source.denominator, 1)
