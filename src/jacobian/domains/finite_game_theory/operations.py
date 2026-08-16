"""Domain adapter for finite game theory operations."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations

from sympy import Matrix

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.finite_game_theory import (
    BestResponseResult,
    NashEquilibriumResult,
    ZeroSumGameRequest,
)


def _format_rational(value: Fraction) -> str:
    if value.denominator == 1:
        return format_canonical_integer(value.numerator)
    return (
        f"{format_canonical_integer(value.numerator)}/"
        f"{format_canonical_integer(value.denominator)}"
    )


def _payoff_matrix(request: ZeroSumGameRequest) -> list[list[Fraction]]:
    matrix = request.payoff_matrix
    entries = [entry.as_fraction() for entry in matrix.entries]
    return [
        [entries[row * matrix.n_cols + col] for col in range(matrix.n_cols)]
        for row in range(matrix.n_rows)
    ]


def compute_best_response(request: ZeroSumGameRequest) -> BestResponseResult:
    """Compute the maximin value and a maximizing row for the row player."""
    matrix = _payoff_matrix(request)
    best_row = 0
    best_value = min(matrix[0])
    for row_index, row in enumerate(matrix[1:], start=1):
        row_min = min(row)
        if row_min > best_value:
            best_value = row_min
            best_row = row_index
    return BestResponseResult(value=_format_rational(best_value), best_row=best_row)


def _try_support(  # noqa: C901
    matrix: list[list[Fraction]],
    rows: Sequence[int],
    cols: Sequence[int],
) -> tuple[list[Fraction], list[Fraction], Fraction] | None:
    row_count = len(matrix)
    col_count = len(matrix[0])
    support_size = len(rows)
    indifference = []
    rhs = []
    first_row = rows[0]
    for row in rows[1:]:
        indifference.append([matrix[row][col] - matrix[first_row][col] for col in cols])
        rhs.append(Fraction(0))
    indifference.append([Fraction(1)] * support_size)
    rhs.append(Fraction(1))
    try:
        solved = Matrix(indifference).solve(Matrix(rhs))
    except Exception:
        return None
    q_support = [Fraction(solved[index]) for index in range(support_size)]
    if any(weight <= 0 for weight in q_support):
        return None
    q = [Fraction(0)] * col_count
    for index, col in enumerate(cols):
        q[col] = q_support[index]
    row_payoffs = [
        sum(matrix[row][col] * q[col] for col in range(col_count))
        for row in range(row_count)
    ]
    value = row_payoffs[first_row]
    if any(row_payoffs[row] != value for row in rows):
        return None
    if any(row_payoffs[row] > value for row in range(row_count) if row not in rows):
        return None

    column_system = []
    column_rhs = []
    first_col = cols[0]
    for col in cols[1:]:
        column_system.append(
            [matrix[row][col] - matrix[row][first_col] for row in rows]
        )
        column_rhs.append(Fraction(0))
    column_system.append([Fraction(1)] * support_size)
    column_rhs.append(Fraction(1))
    try:
        solved = Matrix(column_system).solve(Matrix(column_rhs))
    except Exception:
        return None
    p_support = [Fraction(solved[index]) for index in range(support_size)]
    if any(weight <= 0 for weight in p_support):
        return None
    p = [Fraction(0)] * row_count
    for index, row in enumerate(rows):
        p[row] = p_support[index]
    col_payoffs = [
        sum(p[row] * matrix[row][col] for row in range(row_count))
        for col in range(col_count)
    ]
    if any(col_payoffs[col] != value for col in cols):
        return None
    if any(col_payoffs[col] < value for col in range(col_count) if col not in cols):
        return None
    return p, q, value


def compute_nash_equilibrium(request: ZeroSumGameRequest) -> NashEquilibriumResult:
    """Compute a mixed Nash equilibrium of a 2-player zero-sum game."""
    matrix = _payoff_matrix(request)
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    for support_size in range(1, min(n_rows, n_cols) + 1):
        for rows in combinations(range(n_rows), support_size):
            for cols in combinations(range(n_cols), support_size):
                solved = _try_support(matrix, rows, cols)
                if solved is None:
                    continue
                row_strategy, col_strategy, value = solved
                return NashEquilibriumResult(
                    row_strategy=tuple(
                        _format_rational(weight) for weight in row_strategy
                    ),
                    col_strategy=tuple(
                        _format_rational(weight) for weight in col_strategy
                    ),
                    value=_format_rational(value),
                )
    raise RuntimeError("zero-sum game has no mixed Nash equilibrium")


__all__ = ["compute_best_response", "compute_nash_equilibrium"]
