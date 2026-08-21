"""Exact Newton interpolation kernels over canonical rationals."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.polynomial_interpolation_ops._kernel import (
    divided_difference_coefficients,
    evaluate_newton_form,
)
from jacobian.math.polynomial_interpolation_ops._models import (
    DividedDifferencesRequest,
    DividedDifferencesResult,
    NewtonEvaluateRequest,
    NewtonEvaluateResult,
    NewtonForm,
    NewtonFormRequest,
)


def _canonical(values: tuple[Fraction, ...]) -> tuple[CanonicalRational, ...]:
    return tuple(CanonicalRational.from_fraction(value) for value in values)


def compute_divided_differences(
    request: DividedDifferencesRequest,
) -> DividedDifferencesResult:
    coefficients = divided_difference_coefficients(
        request.samples.nodes,
        request.samples.values,
    )
    return DividedDifferencesResult(coefficients=_canonical(coefficients))


def compute_newton_form(request: NewtonFormRequest) -> NewtonForm:
    coefficients = divided_difference_coefficients(
        request.samples.nodes,
        request.samples.values,
    )
    return NewtonForm(
        coefficients=_canonical(coefficients),
        nodes=request.samples.nodes,
    )


def compute_newton_evaluate(request: NewtonEvaluateRequest) -> NewtonEvaluateResult:
    return NewtonEvaluateResult(
        result=CanonicalRational.from_fraction(
            evaluate_newton_form(
                request.newton_form.nodes,
                request.newton_form.coefficients,
                request.evaluation_point,
            )
        )
    )


__all__ = [
    "compute_divided_differences",
    "compute_newton_evaluate",
    "compute_newton_form",
]
