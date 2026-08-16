"""Typed contracts for finite game theory operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational

_MAX_ACTIONS = 16
_MAX_RATIONAL_DIGITS = 256


class RationalBimatrixGame(ContractModel):
    """A two-player normal-form game with rational payoffs."""

    row_actions: tuple[str, ...] = Field(min_length=1, max_length=_MAX_ACTIONS)
    column_actions: tuple[str, ...] = Field(min_length=1, max_length=_MAX_ACTIONS)
    row_payoff: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1, max_length=_MAX_ACTIONS,
    )
    column_payoff: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1, max_length=_MAX_ACTIONS,
    )

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        if len(set(self.row_actions)) != len(self.row_actions):
            raise ValueError("row action labels must be unique")
        if len(set(self.column_actions)) != len(self.column_actions):
            raise ValueError("column action labels must be unique")
        m, n = len(self.row_actions), len(self.column_actions)
        if len(self.row_payoff) != m:
            raise ValueError("row_payoff must have row_actions rows")
        for row in self.row_payoff:
            if len(row) != n:
                raise ValueError("row_payoff rows must match column_actions")
        if len(self.column_payoff) != m:
            raise ValueError("column_payoff must have row_actions rows")
        for row in self.column_payoff:
            if len(row) != n:
                raise ValueError("column_payoff rows must match column_actions")
        return self


class MixedStrategy(ContractModel):
    """A mixed strategy (probability distribution over actions)."""

    probabilities: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_distribution(self) -> Self:
        from fractions import Fraction
        total = Fraction(0)
        for p in self.probabilities:
            total += p.as_fraction()
            if p.as_fraction() < 0:
                raise ValueError("mixed strategy probabilities must be non-negative")
        if total != 1:
            raise ValueError("mixed strategy probabilities must sum to 1")
        return self


class BestResponseRequest(ContractModel):
    """Request to compute the best response for the row player."""

    game: RationalBimatrixGame
    player: Literal["row", "column"]
    opponent_strategy: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        n_opp = len(self.opponent_strategy)
        if self.player == "row":
            if n_opp != len(self.game.column_actions):
                raise ValueError("opponent strategy must match column action count")
        else:
            if n_opp != len(self.game.row_actions):
                raise ValueError("opponent strategy must match row action count")
        return self


class BestResponseResult(ContractModel):
    """Result of a best-response computation."""

    best_actions_indices: tuple[StrictInt, ...] = Field(min_length=1)
    expected_payoff: CanonicalRational
    detail: str = Field(min_length=1, max_length=1024)


class ZeroSumNashRequest(ContractModel):
    """Request to compute a Nash equilibrium of a zero-sum game."""

    payoff_matrix: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1, max_length=_MAX_ACTIONS,
    )
    row_actions: tuple[str, ...] = Field(min_length=1, max_length=_MAX_ACTIONS)
    column_actions: tuple[str, ...] = Field(min_length=1, max_length=_MAX_ACTIONS)

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        m = len(self.row_actions)
        n = len(self.column_actions)
        if len(self.payoff_matrix) != m:
            raise ValueError("payoff matrix must have row_actions rows")
        for row in self.payoff_matrix:
            if len(row) != n:
                raise ValueError("payoff matrix rows must match column_actions")
        return self


class ZeroSumNashResult(ContractModel):
    """Result of computing a Nash equilibrium for a zero-sum game."""

    row_strategy: tuple[CanonicalRational, ...]
    column_strategy: tuple[CanonicalRational, ...]
    game_value: CanonicalRational
    detail: str = Field(min_length=1, max_length=1024)
