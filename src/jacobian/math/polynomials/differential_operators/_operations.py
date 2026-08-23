"""Catalog adapter for differential-operator application."""

from __future__ import annotations

from jacobian.math.polynomials.differential_operators._bounds import (
    validate_application_envelope,
)
from jacobian.math.polynomials.differential_operators._flint import apply_with_flint
from jacobian.math.polynomials.differential_operators._models import (
    DifferentialOperatorApplyRequest,
    DifferentialOperatorApplyResult,
)


def compute_differential_operator_application(
    request: DifferentialOperatorApplyRequest,
) -> DifferentialOperatorApplyResult:
    """Apply the admitted operator power and return its source-bound result."""

    envelope = validate_application_envelope(
        request.polynomial,
        request.operator,
        request.iterations,
        request.expected,
    )
    output = apply_with_flint(
        request.polynomial,
        request.operator,
        request.iterations,
        envelope,
    )
    return DifferentialOperatorApplyResult(
        polynomial=request.polynomial,
        operator=request.operator,
        iterations=request.iterations,
        expected=request.expected,
        output=output,
        is_zero=not output.polynomial.terms,
        matches_expected=(
            None if request.expected is None else output == request.expected
        ),
    )


__all__ = ["compute_differential_operator_application"]
