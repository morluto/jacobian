"""Domain-owned finite game theory operations."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations

from sympy import Matrix

from jacobian.canonical import format_canonical_integer
from jacobian.math.finite_game_theory._models import (
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


def _solve_positive_weights(
    equations: list[list[Fraction]],
    rhs: list[Fraction],
) -> list[Fraction] | None:
    try:
        solved = Matrix(equations).solve(Matrix(rhs))
    except Exception:
        return None
    weights = [Fraction(solved[index]) for index in range(len(rhs))]
    if any(weight <= 0 for weight in weights):
        return None
    return weights


def _embed_weights(
    support: Sequence[int],
    weights: Sequence[Fraction],
    size: int,
) -> list[Fraction]:
    embedded = [Fraction(0)] * size
    for index, position in enumerate(support):
        embedded[position] = weights[index]
    return embedded


def _column_mix_from_rows(
    matrix: list[list[Fraction]],
    rows: Sequence[int],
    cols: Sequence[int],
) -> list[Fraction] | None:
    first_row = rows[0]
    equations = [
        [matrix[row][col] - matrix[first_row][col] for col in cols] for row in rows[1:]
    ]
    rhs = [Fraction(0)] * (len(rows) - 1)
    equations.append([Fraction(1)] * len(cols))
    rhs.append(Fraction(1))
    weights = _solve_positive_weights(equations, rhs)
    if weights is None:
        return None
    return _embed_weights(cols, weights, len(matrix[0]))


def _row_mix_from_cols(
    matrix: list[list[Fraction]],
    rows: Sequence[int],
    cols: Sequence[int],
) -> list[Fraction] | None:
    first_col = cols[0]
    equations = [
        [matrix[row][col] - matrix[row][first_col] for row in rows] for col in cols[1:]
    ]
    rhs = [Fraction(0)] * (len(cols) - 1)
    equations.append([Fraction(1)] * len(rows))
    rhs.append(Fraction(1))
    weights = _solve_positive_weights(equations, rhs)
    if weights is None:
        return None
    return _embed_weights(rows, weights, len(matrix))


def _try_support(
    matrix: list[list[Fraction]],
    rows: Sequence[int],
    cols: Sequence[int],
) -> tuple[list[Fraction], list[Fraction], Fraction] | None:
    q = _column_mix_from_rows(matrix, rows, cols)
    if q is None:
        return None
    row_payoffs = [
        sum(matrix[row][col] * q[col] for col in range(len(q)))
        for row in range(len(matrix))
    ]
    value = Fraction(row_payoffs[rows[0]])
    if any(row_payoffs[row] != value for row in rows):
        return None
    if any(row_payoffs[row] > value for row in range(len(matrix)) if row not in rows):
        return None
    p = _row_mix_from_cols(matrix, rows, cols)
    if p is None:
        return None
    col_payoffs = [
        sum(p[row] * matrix[row][col] for row in range(len(p)))
        for col in range(len(matrix[0]))
    ]
    if any(col_payoffs[col] != value for col in cols):
        return None
    if any(
        col_payoffs[col] < value for col in range(len(matrix[0])) if col not in cols
    ):
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
