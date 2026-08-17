"""Domain adapter for polynomial interpolation operations."""

from __future__ import annotations

from jacobian.contracts.exact import CanonicalRational
from jacobian.math.polynomial_interpolation import (
    MultipointEvaluationRequest,
    MultipointEvaluationResult,
    NewtonInterpolationRequest,
    NewtonInterpolationResult,
    multipoint_evaluate,
    newton_interpolation,
)


def compute_newton_interpolation(
    request: NewtonInterpolationRequest,
) -> NewtonInterpolationResult:
    points = [(p.x.as_fraction(), p.y.as_fraction()) for p in request.points]
    coeffs, div_diffs = newton_interpolation(points)
    return NewtonInterpolationResult(
        coefficients=tuple(CanonicalRational.from_fraction(c) for c in coeffs),
        divided_differences=tuple(
            CanonicalRational.from_fraction(d) for d in div_diffs
        ),
    )


def compute_multipoint_evaluate(
    request: MultipointEvaluationRequest,
) -> MultipointEvaluationResult:
    coeffs = [c.as_fraction() for c in request.coefficients]
    points = [p.as_fraction() for p in request.evaluation_points]
    values = multipoint_evaluate(coeffs, points)
    return MultipointEvaluationResult(
        values=tuple(CanonicalRational.from_fraction(v) for v in values)
    )
