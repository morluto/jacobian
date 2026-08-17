"""Exact bounded recurrence and rational-series producers backed by SymPy."""

from __future__ import annotations

from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.combinatorics._models import (
    IndexedRationalValue,
    LinearRecurrenceEvaluationRequest,
    LinearRecurrenceEvaluationResult,
    PolynomialCoefficientRecurrenceEvaluationRequest,
    PolynomialCoefficientRecurrenceEvaluationResult,
    RationalGeneratingFunctionCoefficientsRequest,
    RationalGeneratingFunctionCoefficientsResult,
)


def _sympy_rational(value: CanonicalRational) -> Any:
    from sympy import Rational

    return Rational(value.num, value.den)


def _wire(value: Any) -> CanonicalRational:
    return CanonicalRational(
        num=format_canonical_integer(int(value.p)),
        den=format_canonical_integer(int(value.q)),
    )


def evaluate_linear_recurrence(
    request: LinearRecurrenceEvaluationRequest,
) -> LinearRecurrenceEvaluationResult:
    """Materialize the complete bounded replay prefix and requested projection."""

    value = request
    requested_indices = (
        tuple(range(value.term_count))
        if value.scope == "PREFIX" and value.term_count is not None
        else value.indices
    )
    replay_scope_end = requested_indices[-1]
    coefficients = tuple(_sympy_rational(item) for item in value.coefficients)
    replay = [
        _sympy_rational(item) for item in value.initial_values[: replay_scope_end + 1]
    ]
    while len(replay) <= replay_scope_end:
        replay.append(
            sum(
                (
                    coefficient * replay[len(replay) - offset]
                    for offset, coefficient in enumerate(coefficients, start=1)
                ),
                start=coefficients[0] * 0,
            )
        )
    replay_wire = tuple(_wire(item) for item in replay)
    return LinearRecurrenceEvaluationResult(
        coefficient_convention=value.coefficient_convention,
        scope=value.scope,
        values=tuple(
            IndexedRationalValue(index=index, value=replay_wire[index])
            for index in requested_indices
        ),
        replay_prefix=replay_wire,
        replay_scope_end=replay_scope_end,
    )


def evaluate_polynomial_coefficient_recurrence(
    request: PolynomialCoefficientRecurrenceEvaluationRequest,
) -> PolynomialCoefficientRecurrenceEvaluationResult:
    """Evaluate and expose residuals for a bounded P-recursive relation."""

    requested_indices = (
        tuple(range(request.term_count))
        if request.scope == "PREFIX" and request.term_count is not None
        else request.indices
    )
    end = requested_indices[-1]
    polynomials = tuple(
        tuple(_sympy_rational(item) for item in polynomial)
        for polynomial in request.coefficient_polynomials
    )
    order = len(polynomials) - 1

    def evaluate_polynomial(polynomial: tuple[Any, ...], index: int) -> Any:
        return sum(
            (
                coefficient * index**power
                for power, coefficient in enumerate(polynomial)
            ),
            start=polynomial[0] * 0,
        )

    replay = [_sympy_rational(item) for item in request.initial_values[: end + 1]]
    residuals: list[IndexedRationalValue] = []
    while len(replay) <= end:
        index = len(replay)
        coefficients = tuple(
            evaluate_polynomial(polynomial, index) for polynomial in polynomials
        )
        replay.append(
            -sum(
                (
                    coefficients[offset] * replay[index - offset]
                    for offset in range(1, order + 1)
                ),
                start=coefficients[0] * 0,
            )
            / coefficients[0]
        )
        residual = sum(
            (
                coefficients[offset] * replay[index - offset]
                for offset in range(order + 1)
            ),
            start=coefficients[0] * 0,
        )
        residuals.append(IndexedRationalValue(index=index, value=_wire(residual)))
    replay_wire = tuple(_wire(item) for item in replay)
    return PolynomialCoefficientRecurrenceEvaluationResult(
        coefficient_convention=request.coefficient_convention,
        polynomial_convention=request.polynomial_convention,
        scope=request.scope,
        recurrence_order=order,
        values=tuple(
            IndexedRationalValue(index=index, value=replay_wire[index])
            for index in requested_indices
        ),
        replay_prefix=replay_wire,
        residuals=tuple(residuals),
        replay_scope_end=end,
    )


def compute_rational_generating_function_coefficients(
    request: RationalGeneratingFunctionCoefficientsRequest,
) -> RationalGeneratingFunctionCoefficientsResult:
    """Expand N(x)/D(x) by exact coefficient recurrence through x^(k-1)."""

    value = request
    numerator = tuple(_sympy_rational(item) for item in value.numerator)
    denominator = tuple(_sympy_rational(item) for item in value.denominator)
    zero = denominator[0] * 0
    coefficients: list[Any] = []
    for degree in range(value.truncation_order):
        numerator_coefficient = numerator[degree] if degree < len(numerator) else zero
        known = sum(
            (
                denominator[offset] * coefficients[degree - offset]
                for offset in range(1, min(degree, len(denominator) - 1) + 1)
            ),
            start=zero,
        )
        coefficients.append((numerator_coefficient - known) / denominator[0])
    residuals = tuple(
        sum(
            (
                denominator[offset] * coefficients[degree - offset]
                for offset in range(min(degree, len(denominator) - 1) + 1)
            ),
            start=zero,
        )
        - (numerator[degree] if degree < len(numerator) else zero)
        for degree in range(value.truncation_order)
    )
    return RationalGeneratingFunctionCoefficientsResult(
        coefficient_convention=value.coefficient_convention,
        expansion_point=value.expansion_point,
        truncation_order=value.truncation_order,
        coefficients=tuple(_wire(item) for item in coefficients),
        residual_congruence=(
            "DENOMINATOR_TIMES_SERIES_MINUS_NUMERATOR_IS_ZERO_MOD_X_TO_ORDER"
        ),
        residual_coefficients=tuple(_wire(item) for item in residuals),
    )


__all__ = [
    "compute_rational_generating_function_coefficients",
    "evaluate_linear_recurrence",
    "evaluate_polynomial_coefficient_recurrence",
]
