"""Native exact operations for rational linear systems."""

from fractions import Fraction
from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.matrices.rational_linear._models import LinearRationalSystem

__all__ = ["inconsistency_witness", "solve"]


def _fmpq(value: Fraction, flint: Any) -> Any:
    return flint.fmpq(value.numerator, value.denominator)


def _canonical_rational(value: Any) -> CanonicalRational:
    return CanonicalRational(
        num=format_canonical_integer(int(value.numerator)),
        den=format_canonical_integer(int(value.denominator)),
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


def _fractions(
    system: LinearRationalSystem,
) -> tuple[list[list[Fraction]], list[Fraction]]:
    coefficients = [
        [value.as_fraction() for value in row] for row in system.coefficients.entries
    ]
    bounds = [value.as_fraction() for value in system.rhs]
    return coefficients, bounds


def solve(system: LinearRationalSystem) -> tuple[CanonicalRational, ...] | None:
    """Return one exact solution, or ``None`` when the system is inconsistent."""

    coefficients, bounds = _fractions(system)
    import flint

    values = _solve(coefficients, bounds, flint)
    if values is None:
        return None
    return tuple(_canonical_rational(value) for value in values)


def inconsistency_witness(
    system: LinearRationalSystem,
) -> tuple[tuple[CanonicalRational, ...], CanonicalRational] | None:
    """Return a separating left witness and nonzero RHS pairing, if any."""

    coefficients, bounds = _fractions(system)
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
        return None
    witness = tuple(_canonical_rational(value) for value in values)
    pairing: Fraction = sum(
        (
            bound * coordinate.as_fraction()
            for bound, coordinate in zip(bounds, witness, strict=True)
        ),
        Fraction(0),
    )
    return witness, CanonicalRational(
        num=format_canonical_integer(pairing.numerator),
        den=format_canonical_integer(pairing.denominator),
    )
