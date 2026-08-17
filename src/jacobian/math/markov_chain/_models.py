"""Typed wire contracts for Markov chain operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel


class TransitionMatrixRequest(StrictModel):
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


class StationaryDistributionResult(StrictModel):
    distribution: tuple[CanonicalRational, ...]
    method: Literal["SYMPY_EIGENVECTOR"] = "SYMPY_EIGENVECTOR"


class ErgodicDecisionResult(StrictModel):
    is_ergodic: bool
    is_irreducible: bool
    is_aperiodic: bool
