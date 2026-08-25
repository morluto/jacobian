"""In-process Python-FLINT rational-linear operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.matrices.rational_linear._models import (
    LinearRationalInconsistencyFindRequest,
    LinearRationalInconsistencyResult,
    LinearRationalSolutionFindRequest,
    LinearRationalSolutionResult,
)


def _fmpq(value: Fraction, flint: Any) -> Any:
    return flint.fmpq(value.numerator, value.denominator)


def _canonical_rational(value: Any) -> CanonicalRational:
    """Convert a backend-native fmpq through the canonical wire representation."""

    return CanonicalRational(
        num=format_canonical_integer(int(value.numerator)),
        den=format_canonical_integer(int(value.denominator)),
    )


def _canonical_fraction(value: Fraction) -> CanonicalRational:
    """Convert one exact Python fraction through the canonical wire form."""

    return CanonicalRational(
        num=format_canonical_integer(value.numerator),
        den=format_canonical_integer(value.denominator),
    )


def _solve(
    coefficients: list[list[Fraction]], rhs: list[Fraction], flint: Any
) -> list[Any] | None:
    augmented = flint.fmpq_mat(
        [
            [_fmpq(value, flint) for value in row] + [_fmpq(bound, flint)]
            for row, bound in zip(coefficients, rhs, strict=True)
        ]
    )
    reduced, _ = augmented.rref()
    columns = len(coefficients[0])
    values = [flint.fmpq(0) for _ in range(columns)]
    for row in range(reduced.nrows()):
        pivot = next(
            (column for column in range(columns) if reduced[row, column] != 0),
            None,
        )
        if pivot is None:
            if reduced[row, columns] != 0:
                return None
            continue
        values[pivot] = reduced[row, columns]
    return values


def compute_rational_solution(
    request: LinearRationalSolutionFindRequest,
) -> LinearRationalSolutionResult:
    system = request.system
    coefficients = [
        [value.as_fraction() for value in row] for row in system.coefficients.entries
    ]
    bounds = [value.as_fraction() for value in system.rhs]
    import flint

    values = _solve(coefficients, bounds, flint)
    if values is None:
        return LinearRationalSolutionResult(system=system, status="INCONSISTENT")
    return LinearRationalSolutionResult(
        system=system,
        values=tuple(_canonical_rational(value) for value in values),
    )


def compute_rational_inconsistency(
    request: LinearRationalInconsistencyFindRequest,
) -> LinearRationalInconsistencyResult:
    system = request.system
    coefficients = [
        [value.as_fraction() for value in row] for row in system.coefficients.entries
    ]
    bounds = [value.as_fraction() for value in system.rhs]
    import flint

    row_count = len(coefficients)
    column_count = len(coefficients[0])
    dual = [
        [coefficients[row][column] for row in range(row_count)]
        for column in range(column_count)
    ]
    dual.append(bounds)
    values = _solve(dual, [Fraction(0)] * column_count + [Fraction(1)], flint)
    if values is None:
        return LinearRationalInconsistencyResult(system=system, status="CONSISTENT")
    witness = tuple(_canonical_rational(value) for value in values)
    pairing: Fraction = sum(
        (
            bound * coordinate.as_fraction()
            for bound, coordinate in zip(bounds, witness, strict=True)
        ),
        Fraction(0),
    )
    return LinearRationalInconsistencyResult(
        system=system,
        left_witness=witness,
        rhs_pairing=_canonical_fraction(pairing),
    )


def rational_linear_operations() -> MathTools:
    """Return the two direct rational-linear operations."""

    return (
        MathTool(
            operation_id="linear.rational_solution.compute",
            title="Compute an exact rational solution",
            description="Return an exact rational solution or an inconsistent outcome.",
            request_type=LinearRationalSolutionFindRequest,
            result_type=LinearRationalSolutionResult,
            run=compute_rational_solution,
            tags=("linear-algebra", "rational", "solution", "exact"),
            examples=(
                example(
                    "identity_solution",
                    "Solve a one-variable identity system.",
                    {
                        "system": {
                            "variables": ["x"],
                            "coefficients": {"entries": [[{"num": "1", "den": "1"}]]},
                            "rhs": [{"num": "2", "den": "1"}],
                        }
                    },
                ),
            ),
        ),
        MathTool(
            operation_id="linear.rational_inconsistency.compute",
            title="Compute an exact rational inconsistency witness",
            description="Return an inconsistency witness or a consistent outcome.",
            request_type=LinearRationalInconsistencyFindRequest,
            result_type=LinearRationalInconsistencyResult,
            run=compute_rational_inconsistency,
            tags=("linear-algebra", "rational", "inconsistency", "exact"),
            examples=(
                example(
                    "contradictory_one_variable_system",
                    "Find a witness for x=0 together with x=1.",
                    {
                        "system": {
                            "variables": ["x"],
                            "coefficients": {
                                "entries": [
                                    [{"num": "1", "den": "1"}],
                                    [{"num": "1", "den": "1"}],
                                ]
                            },
                            "rhs": [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                        }
                    },
                ),
            ),
        ),
    )


__all__ = [
    "compute_rational_inconsistency",
    "compute_rational_solution",
    "rational_linear_operations",
]
