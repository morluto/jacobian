"""Domain-owned finite game theory operations."""

from __future__ import annotations

from fractions import Fraction
from math import lcm

from jacobian._exact import CanonicalRational
from jacobian.math.finite_game_theory._models import (
    BestResponseResult,
    NashEquilibriumResult,
    ZeroSumGameRequest,
)


def _wire_rational(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(value)


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
    return BestResponseResult(value=_wire_rational(best_value), best_row=best_row)


def compute_nash_equilibrium(request: ZeroSumGameRequest) -> NashEquilibriumResult:
    """Compute one exact saddle point of a finite 2-player zero-sum game."""

    import sympy
    from sympy.solvers.simplex import lpmax, lpmin

    matrix = _payoff_matrix(request)
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    denominator_scale = lcm(*(value.denominator for row in matrix for value in row))
    integer_matrix = [
        [int(value * denominator_scale) for value in row] for row in matrix
    ]
    minimum_payoff = min(min(row) for row in integer_matrix)
    shift = max(0, 1 - minimum_payoff)
    shifted_matrix = [[value + shift for value in row] for row in integer_matrix]
    row_symbols = sympy.symbols(f"_row0:{n_rows}")
    column_symbols = sympy.symbols(f"_column0:{n_cols}")

    row_constraints = [symbol >= 0 for symbol in row_symbols]
    row_constraints.extend(
        sum(
            row_symbols[row] * sympy.Rational(shifted_matrix[row][column])
            for row in range(n_rows)
        )
        >= 1
        for column in range(n_cols)
    )
    row_total, row_solution = lpmin(sum(row_symbols), row_constraints)

    column_constraints = [symbol >= 0 for symbol in column_symbols]
    column_constraints.extend(
        sum(
            sympy.Rational(shifted_matrix[row][column]) * column_symbols[column]
            for column in range(n_cols)
        )
        <= 1
        for row in range(n_rows)
    )
    column_total, column_solution = lpmax(sum(column_symbols), column_constraints)
    if row_total != column_total or row_total <= 0:
        raise RuntimeError("exact primal and dual scaled game values disagree")

    row_scale = Fraction(row_total)
    column_scale = Fraction(column_total)
    row_strategy = [
        Fraction(row_solution.get(symbol, 0)) / row_scale for symbol in row_symbols
    ]
    column_strategy = [
        Fraction(column_solution.get(symbol, 0)) / column_scale
        for symbol in column_symbols
    ]
    value = (Fraction(1, 1) / row_scale - shift) / denominator_scale
    if sum(row_strategy) != 1 or any(weight < 0 for weight in row_strategy):
        raise RuntimeError("SymPy returned an invalid row strategy")
    if sum(column_strategy) != 1 or any(weight < 0 for weight in column_strategy):
        raise RuntimeError("SymPy returned an invalid column strategy")
    if any(
        sum(row_strategy[row] * matrix[row][column] for row in range(n_rows)) < value
        for column in range(n_cols)
    ):
        raise RuntimeError("row strategy does not attain the reported game value")
    if any(
        sum(matrix[row][column] * column_strategy[column] for column in range(n_cols))
        > value
        for row in range(n_rows)
    ):
        raise RuntimeError("column strategy does not attain the reported game value")
    return NashEquilibriumResult(
        row_strategy=tuple(_wire_rational(weight) for weight in row_strategy),
        col_strategy=tuple(_wire_rational(weight) for weight in column_strategy),
        value=_wire_rational(value),
    )


__all__ = ["compute_best_response", "compute_nash_equilibrium"]
