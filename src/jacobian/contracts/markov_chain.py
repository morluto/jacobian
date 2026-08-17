"""Typed wire contracts for Markov chain operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational, require_bounded_rational

MAX_MIXING_DIMENSION = 8
MAX_MIXING_RATIONAL_DIGITS = 32
MAX_MIXING_STEPS = 256


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


class MixingTimeRequest(TransitionMatrixRequest):
    """Request an exact bounded mixing-time search."""

    matrix: tuple[tuple[CanonicalRational, ...], ...] = Field(
        min_length=1,
        max_length=MAX_MIXING_DIMENSION,
    )
    epsilon: CanonicalRational = Field(
        description="Tolerance for the total variation distance; must be a positive rational.",
    )
    max_steps: int = Field(
        default=64,
        ge=1,
        le=MAX_MIXING_STEPS,
        strict=True,
        description="Inclusive upper bound for the exact rational-power search.",
    )

    @model_validator(mode="after")
    def require_bounded_mixing_time_request(self) -> Self:
        for row in self.matrix:
            for value in row:
                require_bounded_rational(
                    value,
                    max_digits=MAX_MIXING_RATIONAL_DIGITS,
                    label="mixing-time transition probability",
                )
        require_bounded_rational(
            self.epsilon,
            max_digits=MAX_MIXING_RATIONAL_DIGITS,
            label="mixing-time epsilon",
        )
        if not 0 < self.epsilon.as_fraction() <= 1:
            raise ValueError("epsilon must lie in (0, 1]")
        return self


class MixingTimeResult(ContractModel):
    status: Literal["FOUND", "NOT_ERGODIC", "BOUND_EXCEEDED"]
    epsilon: CanonicalRational
    max_steps: int = Field(ge=1, le=MAX_MIXING_STEPS, strict=True)
    steps_examined: int = Field(ge=0, le=MAX_MIXING_STEPS, strict=True)
    mixing_time: int | None = Field(
        default=None,
        ge=0,
        le=MAX_MIXING_STEPS,
        strict=True,
    )
    max_total_variation_distance: CanonicalRational | None = None
    method: Literal["SYMPY_EXACT"] = "SYMPY_EXACT"

    @model_validator(mode="after")
    def bind_search_outcome(self) -> Self:
        epsilon = self.epsilon.as_fraction()
        distance = (
            self.max_total_variation_distance.as_fraction()
            if self.max_total_variation_distance is not None
            else None
        )
        if distance is not None and not 0 <= distance <= 1:
            raise ValueError("total variation distance must lie in [0, 1]")
        if self.status == "FOUND":
            if self.mixing_time is None or distance is None:
                raise ValueError("a found mixing time requires a time and distance")
            if self.steps_examined != self.mixing_time or distance > epsilon:
                raise ValueError("the found mixing time must satisfy epsilon")
        elif self.status == "BOUND_EXCEEDED":
            if self.mixing_time is not None or distance is None:
                raise ValueError("a bound outcome carries only its terminal distance")
            if self.steps_examined != self.max_steps or distance <= epsilon:
                raise ValueError("a bound outcome must remain above epsilon")
        elif (
            self.mixing_time is not None
            or distance is not None
            or self.steps_examined != 0
        ):
            raise ValueError("a non-ergodic chain has no mixing-time search value")
        return self
