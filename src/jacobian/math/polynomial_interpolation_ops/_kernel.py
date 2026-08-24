"""Shared exact kernels for polynomial interpolation contracts and operations."""

from __future__ import annotations

from fractions import Fraction
from math import prod
from typing import TYPE_CHECKING

from jacobian._exact import CanonicalRational

if TYPE_CHECKING:
    from jacobian.math.polynomial_interpolation_ops._models import (
        OrdinaryDerivativeJetTable,
    )


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


def ordinary_derivative_value(
    coefficients: tuple[Fraction, ...],
    node: Fraction,
    derivative_order: int,
) -> Fraction:
    """Evaluate one ordinary derivative of an ascending coefficient tuple."""

    return sum(
        (
            coefficients[degree]
            * prod(range(degree - derivative_order + 1, degree + 1))
            * node ** (degree - derivative_order)
            for degree in range(derivative_order, len(coefficients))
        ),
        start=Fraction(0),
    )


def hermite_interpolation_coefficients(
    table: OrdinaryDerivativeJetTable,
) -> tuple[Fraction, ...]:
    """Solve one admitted ordinary-derivative Hermite system exactly.

    python-flint 0.9 has no derivative-jet interpolation entry point, but its
    maintained ``fmpq_mat.solve`` kernel provides exact fraction-free linear
    algebra. Each row is scaled to integers using the same formula proved by
    request preflight; FLINT therefore expands only the already-admitted
    ``M``-square system and fraction-free minors.
    """

    from flint import fmpq_mat

    multiplicity = sum(len(jet.derivatives) for jet in table.jets)
    matrix_entries: list[int] = []
    right_hand_side: list[int] = []
    ordered_jets = sorted(table.jets, key=lambda item: item.node.as_fraction())
    for jet in ordered_jets:
        node_numerator, node_denominator = jet.node.as_integer_ratio()
        for derivative in jet.derivatives:
            order = derivative.derivative_order
            target_numerator, target_denominator = derivative.value.as_integer_ratio()
            for degree in range(multiplicity):
                if degree < order:
                    matrix_entries.append(0)
                    continue
                falling = prod(range(degree - order + 1, degree + 1))
                matrix_entries.append(
                    falling
                    * node_numerator ** (degree - order)
                    * node_denominator ** (multiplicity - 1 - degree)
                    * target_denominator
                )
            right_hand_side.append(
                target_numerator * node_denominator ** (multiplicity - 1 - order)
            )

    matrix = fmpq_mat(multiplicity, multiplicity, matrix_entries)
    target = fmpq_mat(multiplicity, 1, right_hand_side)
    # python-flint 0.9 documents and accepts the algorithm keyword, while its
    # bundled type stub has not yet added that optional argument.
    solution = matrix.solve(target, algorithm="fflu")  # type: ignore[call-arg]
    return tuple(
        Fraction(int(solution[index, 0].p), int(solution[index, 0].q))
        for index in range(multiplicity)
    )


__all__ = [
    "divided_difference_coefficients",
    "evaluate_newton_form",
    "hermite_interpolation_coefficients",
    "ordinary_derivative_value",
]
