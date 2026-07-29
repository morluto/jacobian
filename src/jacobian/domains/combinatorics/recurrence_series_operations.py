"""Exact bounded recurrence and rational-series producers backed by SymPy."""

from __future__ import annotations

from typing import Any, cast

from jacobian.contracts.combinatorics import (
    IndexedRationalValue,
    LinearRecurrenceEvaluationRequest,
    LinearRecurrenceEvaluationResult,
    RationalGeneratingFunctionCoefficientsRequest,
    RationalGeneratingFunctionCoefficientsResult,
)
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.results import ContractModel


def _sympy_rational(value: CanonicalRational) -> Any:
    from sympy import Rational

    return Rational(int(value.num), int(value.den))


def _wire(value: Any) -> CanonicalRational:
    return CanonicalRational(num=str(value.p), den=str(value.q))


def evaluate_linear_recurrence(request: ContractModel) -> ContractModel:
    """Materialize the complete bounded replay prefix and requested projection."""

    value = cast(LinearRecurrenceEvaluationRequest, request)
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


def compute_rational_generating_function_coefficients(
    request: ContractModel,
) -> ContractModel:
    """Expand N(x)/D(x) by exact coefficient recurrence through x^(k-1)."""

    value = cast(RationalGeneratingFunctionCoefficientsRequest, request)
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
]
