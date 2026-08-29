"""Native exact operations for sparse rational linear systems."""

from fractions import Fraction
from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.matrices.rational_linear._models import LinearRationalSystem

__all__ = ["inconsistency_witness", "solve"]


def _canonical_rational(value: Any) -> CanonicalRational:
    return CanonicalRational(
        num=format_canonical_integer(int(value.numerator)),
        den=format_canonical_integer(int(value.denominator)),
    )


def _solve_sparse(
    entries: dict[tuple[int, int], Fraction],
    *,
    row_count: int,
    column_count: int,
    rhs: tuple[Fraction, ...],
) -> tuple[Any, ...] | None:
    """Return one zero-free-variable solution using SymPy's sparse QQ RREF."""

    from sympy import QQ
    from sympy.polys.matrices import DomainMatrix

    rows: dict[int, dict[int, Any]] = {}
    for (row, column), value in entries.items():
        rows.setdefault(row, {})[column] = QQ(value.numerator, value.denominator)
    for row, value in enumerate(rhs):
        if value:
            rows.setdefault(row, {})[column_count] = QQ(
                value.numerator, value.denominator
            )

    augmented = DomainMatrix(rows, (row_count, column_count + 1), QQ)
    reduced, pivots = augmented.rref()
    if column_count in pivots:
        return None
    values = [QQ.zero] * column_count
    for row, pivot in enumerate(pivots):
        values[pivot] = reduced[row, column_count].element
    return tuple(values)


def _fractions(
    system: LinearRationalSystem,
) -> tuple[dict[tuple[int, int], Fraction], tuple[Fraction, ...]]:
    coefficients = {
        (item.row, item.column): item.value.as_fraction()
        for item in system.coefficients
    }
    return coefficients, tuple(value.as_fraction() for value in system.rhs)


def solve(system: LinearRationalSystem) -> tuple[CanonicalRational, ...] | None:
    """Return one exact solution, or ``None`` when the system is inconsistent."""

    coefficients, bounds = _fractions(system)
    values = _solve_sparse(
        coefficients,
        row_count=system.row_count,
        column_count=len(system.variables),
        rhs=bounds,
    )
    if values is None:
        return None
    return tuple(_canonical_rational(value) for value in values)


def inconsistency_witness(
    system: LinearRationalSystem,
) -> tuple[tuple[CanonicalRational, ...], CanonicalRational] | None:
    """Return a separating left witness and nonzero RHS pairing, if any."""

    coefficients, bounds = _fractions(system)
    dual = {(column, row): value for (row, column), value in coefficients.items()}
    dual.update(
        {
            (len(system.variables), row): value
            for row, value in enumerate(bounds)
            if value
        }
    )
    values = _solve_sparse(
        dual,
        row_count=len(system.variables) + 1,
        column_count=system.row_count,
        rhs=(Fraction(0),) * len(system.variables) + (Fraction(1),),
    )
    if values is None:
        return None
    witness = tuple(_canonical_rational(value) for value in values)
    pairing = sum(
        (
            bound * coordinate.as_fraction()
            for bound, coordinate in zip(bounds, witness, strict=True)
        ),
        Fraction(0),
    )
    return witness, CanonicalRational.from_fraction(pairing)
