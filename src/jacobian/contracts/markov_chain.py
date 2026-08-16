"""Typed wire contracts for Markov chain operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational


class TransitionMatrixRequest(ContractModel):
    """A finite stochastic transition matrix with rational entries."""

    matrix: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1, max_length=32
    )

    @model_validator(mode="after")
    def require_stochastic_square_matrix(self) -> Self:
        dimension = len(self.matrix)
        if any(len(row) != dimension for row in self.matrix):
            raise ValueError("transition matrix must be square")
        for row in self.matrix:
            values = tuple(value.as_fraction() for value in row)
            if any(value < 0 for value in values):
                raise ValueError("transition probabilities must be nonnegative")
            if sum(values) != 1:
                raise ValueError("each transition row must sum to one")
        return self


class StationaryDistributionResult(ContractModel):
    distribution: tuple[CanonicalRational, ...]
    method: Literal["SYMPY_EIGENVECTOR"] = "SYMPY_EIGENVECTOR"


class ErgodicDecisionResult(ContractModel):
    is_ergodic: bool
    is_irreducible: bool
    is_aperiodic: bool


class MixingTimeRequest(ContractModel):
    """Request to compute the exact mixing time of a finite Markov chain."""

    matrix: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1, max_length=32
    )
    epsilon: CanonicalRational = Field(
        description="Tolerance for the total variation distance; must be a positive rational.",
    )
    max_steps: int = Field(
        default=10_000,
        ge=1,
        le=100_000,
        description="Upper bound on the number of steps to search before failing closed.",
    )

    @model_validator(mode="after")
    def require_valid_mixing_time_request(self) -> Self:
        dimension = len(self.matrix)
        if any(len(row) != dimension for row in self.matrix):
            raise ValueError("transition matrix must be square")
        for row in self.matrix:
            values = tuple(value.as_fraction() for value in row)
            if any(value < 0 for value in values):
                raise ValueError("transition probabilities must be nonnegative")
            if sum(values) != 1:
                raise ValueError("each transition row must sum to one")
        eps = self.epsilon.as_fraction()
        if eps <= 0:
            raise ValueError("epsilon must be a positive rational")
        return self


class MixingTimeResult(ContractModel):
    mixing_time: int
    method: Literal["SYMPY_EXACT"] = "SYMPY_EXACT"
