"""Domain-owned finite game theory operations."""

from __future__ import annotations

from fractions import Fraction
from math import lcm

from jacobian._exact import CanonicalRational
from jacobian.math.finite_game_theory._models import (
    BestResponseResult,
    DeterministicTerminalGameRequest,
    NashEquilibriumRequest,
    NashEquilibriumResult,
    ZeroSumGameRequest,
)
from jacobian.math.finite_game_theory.operations import _solve_terminal_game_data
from jacobian.math.finite_game_theory.values import DeterministicTerminalGameSolution


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
    return BestResponseResult._from_kernel(
        value=_wire_rational(best_value), best_row=best_row
    )


def compute_nash_equilibrium(
    request: NashEquilibriumRequest,
) -> NashEquilibriumResult:
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
    return NashEquilibriumResult._from_kernel(
        row_strategy=tuple(_wire_rational(weight) for weight in row_strategy),
        col_strategy=tuple(_wire_rational(weight) for weight in column_strategy),
        value=_wire_rational(value),
    )


def compute_deterministic_terminal_game(
    request: DeterministicTerminalGameRequest,
) -> DeterministicTerminalGameSolution:
    """Compute every value and canonical optimal stationary strategy pair."""

    value_classes, max_strategy, min_strategy = _solve_terminal_game_data(request.game)
    return DeterministicTerminalGameSolution._from_kernel(
        request.game, value_classes, max_strategy, min_strategy
    )


def verify_best_response_result(
    request: ZeroSumGameRequest, result: BestResponseResult
) -> bool:
    """Check one independently supplied maximin-row claim."""

    matrix = _payoff_matrix(request)
    if result.best_row >= len(matrix):
        return False
    value = result.value.as_fraction()
    row_minima = tuple(min(row) for row in matrix)
    return value == max(row_minima) and row_minima[result.best_row] == value


def verify_nash_equilibrium_result(
    request: NashEquilibriumRequest, result: NashEquilibriumResult
) -> bool:
    """Check a zero-sum equilibrium witness inside the admitted LP envelope."""

    matrix = _payoff_matrix(request)
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    if len(result.row_strategy) != n_rows or len(result.col_strategy) != n_cols:
        return False
    row_strategy = tuple(weight.as_fraction() for weight in result.row_strategy)
    col_strategy = tuple(weight.as_fraction() for weight in result.col_strategy)
    value = result.value.as_fraction()
    if (
        sum(row_strategy) != 1
        or sum(col_strategy) != 1
        or any(weight < 0 for weight in (*row_strategy, *col_strategy))
    ):
        return False
    return all(
        sum(row_strategy[row] * matrix[row][column] for row in range(n_rows)) >= value
        for column in range(n_cols)
    ) and all(
        sum(matrix[row][column] * col_strategy[column] for column in range(n_cols))
        <= value
        for row in range(n_rows)
    )


def verify_deterministic_terminal_game_solution(
    result: DeterministicTerminalGameSolution,
) -> bool:
    """Check one independently supplied terminal-game minimax profile.

    ``result.game`` was admitted before result construction, so this bounded
    replay uses exactly the game's published threshold-work envelope.
    """

    value_classes, max_strategy, min_strategy = _solve_terminal_game_data(result.game)
    return (
        result.value_classes == value_classes
        and result.max_strategy == max_strategy
        and result.min_strategy == min_strategy
    )


__all__ = [
    "compute_best_response",
    "compute_deterministic_terminal_game",
    "compute_nash_equilibrium",
    "verify_best_response_result",
    "verify_deterministic_terminal_game_solution",
    "verify_nash_equilibrium_result",
]
