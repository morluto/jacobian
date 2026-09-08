"""Process-boundary tests for killable denominator-derivative GCDs."""

from __future__ import annotations

import json
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
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    sparse_rational_polynomial_to_sympy,
)
from jacobian.math.polynomials.rational_functions.gradient._gcd_process import (
    forced_denominator_derivative_gcds,
    normalize_admitted_fraction,
    source_is_coprime,
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
    assert factors[0].bound.degrees == (32,)
    assert factors[0].bound.total_degree == 32
    assert factors[0].records
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


def test_source_coprimality_worker_timeout_is_killable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian import process

    def timed_out(*args: Any, **kwargs: Any) -> BoundedProcessResult:
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
    with (
        request_execution(started),
        pytest.raises(
            OperationExecutionTimeoutError,
            match="during source coprimality recognition",
        ),
    ):
        bind_request_deadline(started + 5)
        source_is_coprime(source)


def test_normalization_reuses_the_admitted_derivative_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian import process

    captured: dict[str, Any] = {}

    def fake(*args: Any, **kwargs: Any) -> BoundedProcessResult:
        captured["payload"] = json.loads(kwargs["input_bytes"])
        return BoundedProcessResult(
            returncode=0,
            stdout=json.dumps(
                {
                    "numerator": [[0, "-33", "1"]],
                    "denominator": [[34, "1", "1"], [0, "1", "1"]],
                },
                separators=(",", ":"),
            ).encode(),
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        )

    monkeypatch.setattr(process, "run_bounded_process", fake)
    x = symbols("x")
    source = rational_function_from_sympy(1 / (x + 1) ** 33, ("x",))
    numerator = sparse_rational_polynomial_to_sympy(source.numerator, source.variables)
    denominator = sparse_rational_polynomial_to_sympy(
        source.denominator, source.variables
    )
    factor = [
        [*term.exponents, str(term.coefficient.num), str(term.coefficient.den)]
        for term in source.denominator.terms
    ]
    started = monotonic()
    with request_execution(started):
        bind_request_deadline(started + 60)
        normalize_admitted_fraction(
            numerator.diff(0) * denominator - numerator * denominator.diff(0),
            denominator * denominator,
            source.variables,
            tuple(factor),
        )
    assert captured["payload"]["task"] == "normalize"
    assert captured["payload"]["factor"] == factor
