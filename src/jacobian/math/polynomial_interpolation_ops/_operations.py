"""Exact Newton interpolation kernels over canonical rationals."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.polynomial_interpolation_ops._models import (
    DividedDifferencesRequest,
    DividedDifferencesResult,
    NewtonEvaluateRequest,
    NewtonEvaluateResult,
    NewtonForm,
    NewtonFormRequest,
)


def _divided_difference_coefficients(
    nodes: tuple[CanonicalRational, ...],
    values: tuple[CanonicalRational, ...],
) -> tuple[Fraction, ...]:
    node_values = tuple(node.as_fraction() for node in nodes)
    row = [value.as_fraction() for value in values]
    coefficients = [row[0]]
    for width in range(1, len(node_values)):
        row = [
            (row[index + 1] - row[index])
            / (node_values[index + width] - node_values[index])
            for index in range(len(node_values) - width)
        ]
        coefficients.append(row[0])
    return tuple(coefficients)


def _canonical(values: tuple[Fraction, ...]) -> tuple[CanonicalRational, ...]:
    return tuple(CanonicalRational.from_fraction(value) for value in values)


def compute_divided_differences(
    request: DividedDifferencesRequest,
) -> DividedDifferencesResult:
    coefficients = _divided_difference_coefficients(
        request.samples.nodes,
        request.samples.values,
    )
    return DividedDifferencesResult(coefficients=_canonical(coefficients))


def compute_newton_form(request: NewtonFormRequest) -> NewtonForm:
    coefficients = _divided_difference_coefficients(
        request.samples.nodes,
        request.samples.values,
    )
    return NewtonForm(
        coefficients=_canonical(coefficients),
        nodes=request.samples.nodes,
    )


def compute_newton_evaluate(request: NewtonEvaluateRequest) -> NewtonEvaluateResult:
    nodes = tuple(node.as_fraction() for node in request.newton_form.nodes)
    coefficients = tuple(
        coefficient.as_fraction() for coefficient in request.newton_form.coefficients
    )
    point = request.evaluation_point.as_fraction()
    result = coefficients[-1]
    for index in range(len(coefficients) - 2, -1, -1):
        result = coefficients[index] + (point - nodes[index]) * result
    return NewtonEvaluateResult(result=CanonicalRational.from_fraction(result))


__all__ = [
    "compute_divided_differences",
    "compute_newton_evaluate",
    "compute_newton_form",
]
