"""Exact contract tests for rational Hermite reduction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from sympy import Symbol, cancel, diff

from jacobian.math import polynomials
from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
)
from jacobian.math.polynomials.rational_functions._models import (
    HermiteReductionRequest,
    HermiteReductionResult,
)
from jacobian.math.polynomials.rational_functions._operations import (
    compute_hermite_reduction,
)
from jacobian.math.polynomials.rational_functions._tools import TOOLS

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


def test_native_hermite_reduction_uses_canonical_polynomial_owner() -> None:
    function = rational_function_from_sympy(1 / (x - 1) ** 2, ("x",))

    rational_part, remainder = polynomials.hermite_reduction(function)

    assert cancel(rational_function_to_sympy(rational_part) + 1 / (x - 1)) == 0
    assert not remainder.numerator.terms


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


def test_result_rejects_source_mutation() -> None:
    result = compute_hermite_reduction(_request(1 / (x - 1)))
    mutated_source = rational_function_from_sympy(1 / (x - 1) + 1, ("x",))

    with pytest.raises(ValidationError):
        HermiteReductionResult.model_validate(
            {
                **result.model_dump(mode="json"),
                "function": mutated_source.model_dump(mode="json"),
            }
        )


def test_result_rejects_nonzero_additive_constant() -> None:
    result = compute_hermite_reduction(_request(x))
    translated_part = rational_function_from_sympy(x**2 / 2 + 1, ("x",))

    with pytest.raises(ValidationError):
        HermiteReductionResult.model_validate(
            {
                **result.model_dump(mode="json"),
                "rational_part": translated_part.model_dump(mode="json"),
                "rational_primitive": translated_part.model_dump(mode="json"),
            }
        )


def test_result_rejects_non_square_free_remainder() -> None:
    request = _request(1 / (x - 1) ** 2)
    zero = rational_function_from_sympy(0, ("x",))

    with pytest.raises(ValidationError):
        HermiteReductionResult(
            function=request.function,
            rational_part=zero,
            remainder=request.function,
            rational_primitive_status="NO_RATIONAL_PRIMITIVE",
            rational_primitive=None,
        )


def test_result_rejects_improper_remainder() -> None:
    request = _request(0)

    with pytest.raises(ValidationError):
        HermiteReductionResult(
            function=request.function,
            rational_part=rational_function_from_sympy(x**2 / 2, ("x",)),
            remainder=rational_function_from_sympy(-x, ("x",)),
            rational_primitive_status="NO_RATIONAL_PRIMITIVE",
            rational_primitive=None,
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
    with pytest.raises(ValidationError):
        _request(expression)


def test_native_api_rejects_before_backend_result_growth() -> None:
    denominator = 10**128 - 1
    function = rational_function_from_sympy(x**64 / denominator, ("x",))

    with pytest.raises(ValueError, match="Hermite-reduction numerator"):
        polynomials.hermite_reduction(function)


def test_request_rejects_multivariate_function() -> None:
    y = Symbol("y")
    function = rational_function_from_sympy(x + y, ("x", "y"))

    with pytest.raises(ValidationError):
        HermiteReductionRequest(function=function)


def test_example_description_states_input_preconditions() -> None:
    """The published example teaches the canonical QQ(x) wire contract."""

    (tool,) = TOOLS
    assert "envelope admits numerator degree 6" in tool.description.lower()
    for example_spec in tool.examples:
        text = str(example_spec.description).lower()
        assert "canonical univariate qq(x)" in text
        assert "one variable x" in text
        assert "numerator degree" in text
        assert "denominator degree" in text
        assert "two-digit rational coefficient components" in text


def test_request_accepts_repeated_pole_work_envelope_boundary() -> None:
    numerator = sum((99 - exponent) * x**exponent for exponent in range(7))
    result = compute_hermite_reduction(_request(numerator / (x - 3) ** 3))

    rational_part = rational_function_to_sympy(result.rational_part)
    remainder = rational_function_to_sympy(result.remainder)
    assert cancel(diff(rational_part, x) + remainder - numerator / (x - 3) ** 3) == 0
    for value in (result.rational_part, result.remainder):
        for polynomial in (value.numerator, value.denominator):
            for term in polynomial.terms:
                coefficient = term.coefficient
                assert len(coefficient.num.lstrip("-")) <= 128
                assert len(coefficient.den) <= 128
