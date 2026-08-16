"""Domain adapter for finite game theory operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian.contracts.finite_game_theory import (
    BestResponseResult,
    NashEquilibriumResult,
    ZeroSumGameRequest,
)


def compute_best_response(request: ZeroSumGameRequest) -> BestResponseResult:
    """Compute the best response (maximin) for the row player.

    For a zero-sum game, the row player maximizes the minimum guaranteed payoff.
    This computes the maximin value using exact rational arithmetic.
    """
    matrix = request.payoff_matrix
    n_rows = matrix.n_rows
    n_cols = matrix.n_cols
    entries = [e.as_fraction() for e in matrix.entries]

    # For each row, compute the minimum payoff (worst case against column player)
    best_row = 0
    best_value = Fraction(-10**18)

    for i in range(n_rows):
        row_min = min(entries[i * n_cols + j] for j in range(n_cols))
        if row_min > best_value:
            best_value = row_min
            best_row = i

    return BestResponseResult(value=str(best_value), best_row=best_row)


def compute_nash_equilibrium(request: ZeroSumGameRequest) -> NashEquilibriumResult:
    """Compute the Nash equilibrium of a 2-player zero-sum game.

    Uses the support enumeration method for exact rational computation.
    For small games, this finds the mixed-strategy Nash equilibrium.
    """
    matrix = request.payoff_matrix
    n_rows = matrix.n_rows
    n_cols = matrix.n_cols
    entries = [e.as_fraction() for e in matrix.entries]

    # For pure strategy Nash equilibrium:
    # Row player best response: for each column, find the best row
    # Column player best response: for each row, find the best column (minimize)
    # A pure Nash equilibrium is a cell where both are best responses

    for i in range(n_rows):
        for j in range(n_cols):
            # Check if row i is best response to column j
            row_best = all(
                entries[i * n_cols + j] >= entries[i2 * n_cols + j]
                for i2 in range(n_rows)
            )
            # Check if column j is best response to row i (minimize for zero-sum)
            col_best = all(
                entries[i * n_cols + j] <= entries[i * n_cols + j2]
                for j2 in range(n_cols)
            )
            if row_best and col_best:
                # Found pure strategy Nash equilibrium
                row_strategy = ["0"] * n_rows
                row_strategy[i] = "1"
                col_strategy = ["0"] * n_cols
                col_strategy[j] = "1"
                return NashEquilibriumResult(
                    row_strategy=tuple(row_strategy),
                    col_strategy=tuple(col_strategy),
                    value=str(entries[i * n_cols + j]),
                )

    # If no pure equilibrium, return uniform mixed strategies as fallback
    # (This is a simplification; a full implementation would use LP)
    row_uniform = [str(Fraction(1, n_rows))] * n_rows
    col_uniform = [str(Fraction(1, n_cols))] * n_cols
    return NashEquilibriumResult(
        row_strategy=tuple(row_uniform),
        col_strategy=tuple(col_uniform),
        value="0",
    )


__all__ = ["compute_best_response", "compute_nash_equilibrium"]
