"""Domain-owned finite game theory operations."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations

from sympy import Matrix, Rational

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


def _solve_underdetermined_mix(
    equations: list[list[Fraction]],
    rhs: list[Fraction],
    n_positive: int,
) -> tuple[list[Fraction], Fraction] | None:
    """Find a strictly positive rational solution of an underdetermined system.

    The leading ``n_positive`` unknowns must be positive; a trailing value
    (the game value) is returned unconstrained. Searches a small rational grid
    over the free parameters.
    """
    try:
        solved, params = Matrix(equations).gauss_jordan_solve(Matrix(rhs))
    except Exception:
        return None
    if not params:
        return None
    free_syms = solved.free_symbols
    if not free_syms:
        return None
    for d in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16):
        for num in range(1, d):
            candidate = Fraction(num, d)
            subs = {
                s: Rational(candidate.numerator, candidate.denominator)
                for s in free_syms
            }
            try:
                expr = solved.subs(subs)
                vals = [Fraction(expr[i]) for i in range(n_positive + 1)]
            except Exception:
                continue
            if all(v > 0 for v in vals[:n_positive]):
                return vals[:n_positive], vals[n_positive]
    return None


def _solve_column_mix(
    matrix: list[list[Fraction]],
    rows: Sequence[int],
    cols: Sequence[int],
) -> tuple[list[Fraction], Fraction] | None:
    """Find column mix q (over cols) and game value v.

    Constraints: A[row, ·] · q = v for each row in rows, and sum(q) = 1.
    The unknowns are q[0..len(cols)-1] and v.
    """
    n_cols = len(cols)
    equations: list[list[Fraction]] = []
    rhs: list[Fraction] = []
    for row in rows:
        equations.append([Fraction(matrix[row][col]) for col in cols] + [Fraction(-1)])
        rhs.append(Fraction(0))
    equations.append([Fraction(1)] * n_cols + [Fraction(0)])
    rhs.append(Fraction(1))

    try:
        solved = Matrix(equations).solve(Matrix(rhs))
        q = [Fraction(solved[i]) for i in range(n_cols)]
        v = Fraction(solved[n_cols])
    except Exception:
        fallback = _solve_underdetermined_mix(equations, rhs, n_cols)
        if fallback is None:
            return None
        q, v = fallback
    if any(weight <= 0 for weight in q):
        return None
    return q, v


def _solve_row_mix(
    matrix: list[list[Fraction]],
    rows: Sequence[int],
    cols: Sequence[int],
    v: Fraction,
) -> list[Fraction] | None:
    """Find row mix p (over rows) s.t. p·A[·,col] = v for all col in cols, sum(p)=1."""
    n_rows = len(rows)
    equations: list[list[Fraction]] = []
    rhs: list[Fraction] = []
    for col in cols:
        equations.append([Fraction(matrix[row][col]) for row in rows])
        rhs.append(v)
    equations.append([Fraction(1)] * n_rows)
    rhs.append(Fraction(1))

    try:
        solved = Matrix(equations).solve(Matrix(rhs))
        p = [Fraction(solved[i]) for i in range(n_rows)]
    except Exception:
        fallback = _solve_underdetermined_mix(equations, rhs, n_rows)
        if fallback is None:
            return None
        p, _ = fallback
    if any(weight <= 0 for weight in p):
        return None
    return p


def _try_support(
    matrix: list[list[Fraction]],
    rows: Sequence[int],
    cols: Sequence[int],
) -> tuple[list[Fraction], list[Fraction], Fraction] | None:
    n_rows = len(matrix)
    n_cols = len(matrix[0])

    col_result = _solve_column_mix(matrix, rows, cols)
    if col_result is None:
        return None
    q, v = col_result

    q_full = [Fraction(0)] * n_cols
    for i, col in enumerate(cols):
        q_full[col] = q[i]

    row_payoffs = [
        sum(matrix[row][col] * q_full[col] for col in range(n_cols))
        for row in range(n_rows)
    ]
    if any(row_payoffs[row] != v for row in rows):
        return None
    if any(row_payoffs[row] > v for row in range(n_rows) if row not in rows):
        return None

    p = _solve_row_mix(matrix, rows, cols, v)
    if p is None:
        return None

    p_full = [Fraction(0)] * n_rows
    for i, row in enumerate(rows):
        p_full[row] = p[i]

    col_payoffs = [
        sum(p_full[row] * matrix[row][col] for row in range(n_rows))
        for col in range(n_cols)
    ]
    if any(col_payoffs[col] != v for col in cols):
        return None
    if any(col_payoffs[col] < v for col in range(n_cols) if col not in cols):
        return None

    return p_full, q_full, v


def compute_nash_equilibrium(request: ZeroSumGameRequest) -> NashEquilibriumResult:
    """Compute a mixed Nash equilibrium of a 2-player zero-sum game."""
    matrix = _payoff_matrix(request)
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    for total_support in range(2, n_rows + n_cols + 1):
        seen_min = max(1, total_support - n_cols)
        seen_max = min(n_rows, total_support - 1)
        for row_support_size in range(seen_min, seen_max + 1):
            col_support_size = total_support - row_support_size
            for rows in combinations(range(n_rows), row_support_size):
                for cols in combinations(range(n_cols), col_support_size):
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
