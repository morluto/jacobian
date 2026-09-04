"""Native exact operations for sparse rational linear systems."""

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Any, Literal

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.rational_linear._models import LinearRationalSystem

__all__ = ["inconsistency_witness", "solve"]

MAX_LINEAR_SCALAR_WORK = 100_000_000
MAX_FLINT_LINEAR_SCALAR_WORK = 500_000_000
MAX_FLINT_LINEAR_CELLS = 1_000_000


@dataclass(frozen=True)
class _LinearPlan:
    entries: dict[tuple[int, int], Fraction]
    bounds: tuple[Fraction, ...]
    row_count: int
    column_count: int
    backend: str


def _decimal_digits_from_bits(bits: int) -> int:
    return max(1, (bits * 30_103 + 99_999) // 100_000)


def _minor_component_bits(
    rows: tuple[tuple[Fraction, ...], ...], *, column_count: int
) -> int:
    rank_bound = min(len(rows), column_count)
    numerator_bits: list[int] = []
    denominator_bits: list[int] = []
    for row in rows:
        denominator = 1
        for value in row:
            denominator *= value.denominator // gcd(denominator, value.denominator)
            if _decimal_digits_from_bits(denominator.bit_length()) > (
                MAX_CANONICAL_RATIONAL_DIGITS
            ):
                return MAX_CANONICAL_RATIONAL_DIGITS * 4
        largest = max(
            (
                abs(value.numerator) * (denominator // value.denominator)
                for value in row
            ),
            default=0,
        )
        numerator_bits.append(
            largest.bit_length() + (column_count.bit_length() + 1) // 2
        )
        denominator_bits.append(denominator.bit_length())
    return sum(sorted(numerator_bits, reverse=True)[:rank_bound]) + sum(
        sorted(denominator_bits, reverse=True)[:rank_bound]
    )


def _reject(message: str) -> None:
    raise OperationDomainValidationError(
        location=("system",), code="matrix.budget_exceeded", message=message
    )


def _admit_system(
    system: LinearRationalSystem, *, outcome: Literal["solution", "witness"]
) -> _LinearPlan:
    rows = system.coefficients.row_count
    columns = system.coefficients.column_count
    entries = {
        (item.row, item.column): item.value.as_fraction()
        for item in system.coefficients.entries
    }
    bounds = tuple(value.as_fraction() for value in system.rhs)
    scalar_digits = max(
        len(component.lstrip("-"))
        for value in tuple(item.value for item in system.coefficients.entries)
        + system.rhs
        for component in (value.num, value.den)
    )
    if outcome == "solution":
        work = rows * (columns + 1) * min(rows, columns + 1)
        dense_cells = rows * (columns + 1)
        grouped: dict[int, list[Fraction]] = {row: [] for row in range(rows)}
        for (row, _column), fraction in entries.items():
            grouped[row].append(fraction)
        for row, bound in enumerate(bounds):
            if bound:
                grouped[row].append(bound)
        minor_rows = tuple(tuple(grouped[row]) for row in range(rows))
        minor_columns = columns + 1
    else:
        work = (columns + 1) * (rows + 1) * min(columns + 1, rows + 1)
        dense_cells = (columns + 1) * (rows + 1)
        grouped = {column: [] for column in range(columns)}
        for (_row, column), fraction in entries.items():
            grouped[column].append(fraction)
        minor_rows = (
            *(tuple(grouped[column]) for column in range(columns)),
            tuple(bound for bound in bounds if bound),
        )
        minor_columns = rows + 1
    charged_work = work * scalar_digits
    if charged_work > MAX_LINEAR_SCALAR_WORK and (
        charged_work > MAX_FLINT_LINEAR_SCALAR_WORK
        or dense_cells > MAX_FLINT_LINEAR_CELLS
    ):
        _reject(
            "sparse exact linear algebra exceeds the "
            f"{MAX_FLINT_LINEAR_SCALAR_WORK:,}-unit scalar-work budget or "
            f"{MAX_FLINT_LINEAR_CELLS:,}-cell FLINT envelope"
        )

    result_digits = _decimal_digits_from_bits(
        _minor_component_bits(minor_rows, column_count=minor_columns)
    )
    if result_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        _reject("sparse exact linear algebra exceeds the canonical result-height bound")

    return _LinearPlan(
        entries=entries,
        bounds=bounds,
        row_count=rows,
        column_count=columns,
        backend=(
            "sympy_sparse" if charged_work <= MAX_LINEAR_SCALAR_WORK else "flint_dense"
        ),
    )


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


def _solve_flint(
    entries: dict[tuple[int, int], Fraction],
    *,
    row_count: int,
    column_count: int,
    rhs: tuple[Fraction, ...],
) -> tuple[Fraction, ...] | None:
    """Return one zero-free-variable solution through FLINT's dense QQ RREF."""

    from jacobian.math.matrices._flint import rational_rref

    dense = [[Fraction(0) for _ in range(column_count + 1)] for _ in range(row_count)]
    for (row, column), value in entries.items():
        dense[row][column] = value
    for row, value in enumerate(rhs):
        dense[row][column_count] = value
    reduced, rank_value = rational_rref(tuple(tuple(row) for row in dense))
    pivots = tuple(
        next(column for column, value in enumerate(row) if value)
        for row in reduced[:rank_value]
    )
    if column_count in pivots:
        return None
    values = [Fraction(0)] * column_count
    for row, pivot in enumerate(pivots):
        values[pivot] = reduced[row][column_count]
    return tuple(values)


def solve(system: LinearRationalSystem) -> tuple[CanonicalRational, ...] | None:
    """Return one exact solution, or ``None`` when the system is inconsistent."""

    plan = _admit_system(system, outcome="solution")
    kernel = _solve_sparse if plan.backend == "sympy_sparse" else _solve_flint
    values = kernel(
        plan.entries,
        row_count=plan.row_count,
        column_count=plan.column_count,
        rhs=plan.bounds,
    )
    if values is None:
        return None
    return tuple(_canonical_rational(value) for value in values)


def inconsistency_witness(
    system: LinearRationalSystem,
) -> tuple[tuple[CanonicalRational, ...], CanonicalRational] | None:
    """Return a separating left witness and nonzero RHS pairing, if any."""

    plan = _admit_system(system, outcome="witness")
    dual = {(column, row): value for (row, column), value in plan.entries.items()}
    dual.update(
        {
            (len(system.variables), row): value
            for row, value in enumerate(plan.bounds)
            if value
        }
    )
    kernel = _solve_sparse if plan.backend == "sympy_sparse" else _solve_flint
    values = kernel(
        dual,
        row_count=len(system.variables) + 1,
        column_count=plan.row_count,
        rhs=(Fraction(0),) * len(system.variables) + (Fraction(1),),
    )
    if values is None:
        return None
    witness = tuple(_canonical_rational(value) for value in values)
    pairing = sum(
        (
            bound * coordinate.as_fraction()
            for bound, coordinate in zip(plan.bounds, witness, strict=True)
        ),
        Fraction(0),
    )
    return witness, CanonicalRational.from_fraction(pairing)
