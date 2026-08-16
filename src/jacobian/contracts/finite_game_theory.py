"""Typed wire contracts for finite game theory operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational

MAX_PLAYERS = 10
MAX_STRATEGIES = 8


class PayoffMatrix(ContractModel):
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


class ZeroSumGameRequest(ContractModel):
    """A 2-player zero-sum game specified by the row player's payoff matrix."""

    payoff_matrix: PayoffMatrix


class BestResponseResult(ContractModel):
    """Best response values for the row player."""

    value: str
    best_row: int = Field(ge=0)


class NashEquilibriumResult(ContractModel):
    """Nash equilibrium of a 2-player zero-sum game."""

    row_strategy: tuple[str, ...]
    col_strategy: tuple[str, ...]
    value: str


__all__ = [
    "BestResponseResult",
    "NashEquilibriumResult",
    "PayoffMatrix",
    "ZeroSumGameRequest",
]
