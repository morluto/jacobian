"""Exact contract tests for rational Hermite reduction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from sympy import Symbol, cancel, diff

from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
)
from jacobian.math.rational_functions._models import (
    HermiteReductionRequest,
    HermiteReductionResult,
)
from jacobian.math.rational_functions._operations import compute_hermite_reduction

x = Symbol("x")


def _request(expression: object) -> HermiteReductionRequest:
    return HermiteReductionRequest(
        function=rational_function_from_sympy(expression, ("x",))
    )


@pytest.mark.parametrize(
    ("expression", "status"),
    [
        (x**2 + 1, "RATIONAL_PRIMITIVE"),
        (1 / (x - 1) ** 2, "RATIONAL_PRIMITIVE"),
        (1 / (x - 1), "NO_RATIONAL_PRIMITIVE"),
        (1 / (x**2 + 1), "NO_RATIONAL_PRIMITIVE"),
        (1 / ((x - 1) ** 2 * (x + 1)), "NO_RATIONAL_PRIMITIVE"),
    ],
)
def test_hermite_reduction_reconstructs_source(
    expression: object,
    status: str,
) -> None:
    result = compute_hermite_reduction(_request(expression))

    rational_part = rational_function_to_sympy(result.rational_part)
    remainder = rational_function_to_sympy(result.remainder)
    assert cancel(diff(rational_part, x) + remainder - expression) == 0
    assert result.rational_primitive_status == status
    assert (result.rational_primitive is not None) == (status == "RATIONAL_PRIMITIVE")


def test_hermite_reduction_uses_zero_integration_constant() -> None:
    result = compute_hermite_reduction(_request(x**2 + 1))

    assert (
        cancel(rational_function_to_sympy(result.rational_part) - (x**3 / 3 + x)) == 0
    )
    assert not result.remainder.numerator.terms


def test_adding_exact_derivative_preserves_remainder() -> None:
    source = compute_hermite_reduction(_request(1 / (x - 1)))
    translated = compute_hermite_reduction(
        _request(1 / (x - 1) + diff(-1 / (x + 1), x))
    )

    assert translated.remainder == source.remainder
    assert (
        cancel(
            rational_function_to_sympy(translated.rational_part)
            - rational_function_to_sympy(source.rational_part)
            + 1 / (x + 1)
        )
        == 0
    )


def test_result_rejects_mutated_remainder() -> None:
    result = compute_hermite_reduction(_request(1 / ((x - 1) ** 2 * (x + 1))))
    mutated = rational_function_from_sympy(1 / (x + 1), ("x",))

    with pytest.raises(ValidationError, match="does not reconstruct"):
        HermiteReductionResult.model_validate(
            {
                **result.model_dump(mode="json"),
                "remainder": mutated.model_dump(mode="json"),
            }
        )


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        (x**7, "numerator exponent"),
        (1 / (x**4 + 1), "denominator exponent"),
        (100 * x, "numerator coefficient"),
    ],
)
def test_request_rejects_above_conservative_work_envelope(
    expression: object,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _request(expression)


def test_request_rejects_multivariate_function() -> None:
    y = Symbol("y")
    function = rational_function_from_sympy(x + y, ("x", "y"))

    with pytest.raises(ValidationError, match="exactly one variable"):
        HermiteReductionRequest(function=function)


def test_request_accepts_exact_work_envelope_boundary() -> None:
    result = compute_hermite_reduction(_request((99 * x**6 + 1) / (x**3 + 99 * x + 1)))

    assert result.complete is True
