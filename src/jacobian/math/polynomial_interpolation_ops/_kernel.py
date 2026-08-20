"""Shared exact kernels for polynomial interpolation contracts and operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational


def divided_difference_coefficients(
    nodes: tuple[CanonicalRational, ...],
    values: tuple[CanonicalRational, ...],
) -> tuple[Fraction, ...]:
    """Return the exact Newton coefficients for pairwise-distinct nodes."""

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


def evaluate_newton_form(
    nodes: tuple[CanonicalRational, ...],
    coefficients: tuple[CanonicalRational, ...],
    point: CanonicalRational,
) -> Fraction:
    """Evaluate one Newton form exactly with nested multiplication."""

    node_values = tuple(node.as_fraction() for node in nodes)
    coefficient_values = tuple(value.as_fraction() for value in coefficients)
    point_value = point.as_fraction()
    result = coefficient_values[-1]
    for index in range(len(coefficient_values) - 2, -1, -1):
        result = coefficient_values[index] + (point_value - node_values[index]) * result
    return result


__all__ = ["divided_difference_coefficients", "evaluate_newton_form"]
