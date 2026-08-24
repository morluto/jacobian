"""Typed wire contracts for finite game theory operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.finite_game_theory.values import DeterministicTerminalGame

MAX_PLAYERS = 10
MAX_STRATEGIES = 50


class PayoffMatrix(StrictModel):
    """A payoff matrix for a 2-player normal-form game.

    Row player's payoff is stored; column player's payoff is implied
    (for zero-sum games, it is the negative of the row player's payoff).
    """

    n_rows: int = Field(ge=1, le=MAX_STRATEGIES)
    n_cols: int = Field(ge=1, le=MAX_STRATEGIES)
    entries: tuple[CanonicalRational, ...]

    @model_validator(mode="after")
    def require_valid_size(self) -> Self:
        if len(self.entries) != self.n_rows * self.n_cols:
            raise ValueError("entries must have n_rows * n_cols elements")
        return self


class ZeroSumGameRequest(StrictModel):
    """A 2-player zero-sum game specified by the row player's payoff matrix."""

    payoff_matrix: PayoffMatrix

    @model_validator(mode="after")
    def require_bounded_exact_equilibrium(self) -> Self:
        matrix = self.payoff_matrix
        denominator_digits = sum(len(value.den) for value in matrix.entries)
        numerator_digits = max(len(value.num.lstrip("-")) for value in matrix.entries)
        elimination_dimension = max(matrix.n_rows, matrix.n_cols) + 2
        if elimination_dimension * (denominator_digits + numerator_digits) > 32_768:
            raise ValueError("payoffs exceed the exact-equilibrium result budget")
        return self


class BestResponseResult(StrictModel):
    """Best response values for the row player."""

    value: CanonicalRational
    best_row: int = Field(ge=0)


class NashEquilibriumResult(StrictModel):
    """Nash equilibrium of a 2-player zero-sum game."""

    row_strategy: tuple[CanonicalRational, ...]
    col_strategy: tuple[CanonicalRational, ...]
    value: CanonicalRational


class DeterministicTerminalGameRequest(StrictModel):
    """Solve every position of one complete deterministic terminal game."""

    game: DeterministicTerminalGame = Field(
        description=(
            "Materialized owned arena with exact terminal and infinite-play "
            "payoffs; every nonterminal must have an outgoing move."
        )
    )


__all__ = [
    "BestResponseResult",
    "DeterministicTerminalGameRequest",
    "NashEquilibriumResult",
    "PayoffMatrix",
    "ZeroSumGameRequest",
]
