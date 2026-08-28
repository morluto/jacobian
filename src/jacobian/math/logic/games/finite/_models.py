"""Typed wire contracts for finite game theory operations."""

from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.logic.games.finite.values import DeterministicTerminalGame

MAX_PLAYERS = 10
MAX_STRATEGIES = 50
MAX_EXACT_EQUILIBRIUM_WORK = 32_768


class PayoffMatrix(StrictModel):
    """A payoff matrix for a 2-player normal-form game.

    Row player's payoff is stored; column player's payoff is implied
    (for zero-sum games, it is the negative of the row player's payoff).
    """

    n_rows: int = Field(
        ge=1,
        le=MAX_STRATEGIES,
        description="Number of row-player strategies.",
    )
    n_cols: int = Field(
        ge=1,
        le=MAX_STRATEGIES,
        description="Number of column-player strategies.",
    )
    entries: tuple[CanonicalRational, ...] = Field(
        description=(
            "Row-major payoff entries. The tuple has exactly n_rows * n_cols elements."
        )
    )

    @model_validator(mode="after")
    def require_valid_size(self) -> Self:
        if len(self.entries) != self.n_rows * self.n_cols:
            raise PydanticCustomError(
                "finite_game.payoff_matrix_size",
                "entries must have n_rows * n_cols elements",
            )
        return self


class ZeroSumGameRequest(StrictModel):
    """A 2-player zero-sum game specified by the row player's payoff matrix."""

    payoff_matrix: PayoffMatrix = Field(
        description="Row-major zero-sum payoff matrix with n_rows * n_cols entries."
    )


class NashEquilibriumRequest(ZeroSumGameRequest):
    """A zero-sum game admitted for exact primal/dual equilibrium solving."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A 2-player zero-sum game specified by the row player's payoff "
                "matrix. Exact primal and dual linear programs are admitted only "
                "when (max(n_rows, n_cols) + 2) * (sum of payoff denominator "
                "decimal digits + maximum payoff numerator decimal digits) is at "
                f"most {MAX_EXACT_EQUILIBRIUM_WORK}."
            ),
            "x-jacobian-bounds": {
                "max_exact_equilibrium_work": MAX_EXACT_EQUILIBRIUM_WORK,
            },
        }
    )


class BestResponseResult(StrictModel):
    """Best response values for the row player."""

    value: CanonicalRational
    best_row: int = Field(ge=0)

    @classmethod
    def _from_kernel(cls, value: CanonicalRational, best_row: int) -> Self:
        """Construct trusted output from the owner-local exact kernel."""

        return cls.model_construct(value=value, best_row=best_row)


class NashEquilibriumResult(StrictModel):
    """Nash equilibrium of a 2-player zero-sum game."""

    row_strategy: tuple[CanonicalRational, ...]
    col_strategy: tuple[CanonicalRational, ...]
    value: CanonicalRational

    @classmethod
    def _from_kernel(
        cls,
        row_strategy: tuple[CanonicalRational, ...],
        col_strategy: tuple[CanonicalRational, ...],
        value: CanonicalRational,
    ) -> Self:
        """Construct trusted output from the owner-local exact kernel."""

        return cls.model_construct(
            row_strategy=row_strategy,
            col_strategy=col_strategy,
            value=value,
        )


class DeterministicTerminalGameRequest(StrictModel):
    """Solve every position of one complete deterministic terminal game."""

    game: DeterministicTerminalGame = Field(
        description=(
            "Materialized owned arena with exact terminal and infinite-play "
            "payoffs; every nonterminal must have an outgoing move."
        )
    )


__all__ = [
    "MAX_EXACT_EQUILIBRIUM_WORK",
    "BestResponseResult",
    "DeterministicTerminalGameRequest",
    "NashEquilibriumRequest",
    "NashEquilibriumResult",
    "PayoffMatrix",
    "ZeroSumGameRequest",
]
