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
    return DifferentialOperatorApplyResult._from_kernel(request, output)


def verify_differential_operator_application_result(
    result: DifferentialOperatorApplyResult,
) -> bool:
    """Replay one independently supplied result inside its admitted envelope."""

    try:
        envelope = validate_application_envelope(
            result.polynomial,
            result.operator,
            result.iterations,
            result.expected,
        )
        return result.output == apply_with_flint(
            result.polynomial,
            result.operator,
            result.iterations,
            envelope,
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        return False


__all__ = [
    "compute_differential_operator_application",
    "verify_differential_operator_application_result",
]
