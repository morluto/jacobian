"""Typed wire contracts for finite stochastic process operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.probability.stochastic_processes.values import (
    FiniteProbabilitySpace,
    FiniteRandomVariable,
    FiniteSigmaAlgebra,
)

MAX_STOCHASTIC_VALUE_DIGITS = 256


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by finite stochastic contracts."""

    return PydanticCustomError(f"finite_stochastic_process.{reason}", message)


class FromObservationRequest(StrictModel):
    """Construct a sigma algebra from an observation map."""

    space: FiniteProbabilitySpace
    observation: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_observation_matches_space(self) -> Self:
        if len(self.observation) != len(self.space.samples):
            raise _validation_error(
                "observation_length_mismatch",
                "observation must have one entry per sample",
            )
        return self


class JoinRequest(StrictModel):
    """Compute the join of two sigma algebras."""

    sigma1: FiniteSigmaAlgebra
    sigma2: FiniteSigmaAlgebra

    @model_validator(mode="after")
    def require_same_space(self) -> Self:
        if self.sigma1.space != self.sigma2.space:
            raise _validation_error(
                "sigma_space_mismatch",
                "sigma algebras must share the same probability space",
            )
        return self


class ConditionalExpectationRequest(StrictModel):
    """Compute E[X | G]."""

    rv: FiniteRandomVariable
    sigma: FiniteSigmaAlgebra

    @model_validator(mode="after")
    def require_same_space(self) -> Self:
        if self.rv.space != self.sigma.space:
            raise _validation_error(
                "conditional_expectation_space_mismatch",
                "random variable and sigma algebra must share the same probability space",
            )
        return self


class FiltrationRequest(StrictModel):
    """Compute the natural filtration of observations."""

    space: FiniteProbabilitySpace
    observations: tuple[tuple[str, ...], ...] = Field(default=())

    @model_validator(mode="after")
    def require_observations_match_space(self) -> Self:
        for obs in self.observations:
            if len(obs) != len(self.space.samples):
                raise _validation_error(
                    "observation_length_mismatch",
                    "observation must have one entry per sample",
                )
        return self


class DoobMartingaleRequest(StrictModel):
    """Compute the Doob martingale of a payoff process."""

    space: FiniteProbabilitySpace
    observations: tuple[tuple[str, ...], ...] = Field(default=())
    payoff: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_payoff_matches_space(self) -> Self:
        if len(self.payoff) != len(self.space.samples):
            raise _validation_error(
                "payoff_length_mismatch", "payoff must have one entry per sample"
            )
        for obs in self.observations:
            if len(obs) != len(self.space.samples):
                raise _validation_error(
                    "observation_length_mismatch",
                    "observation must have one entry per sample",
                )
        return self


class FiltrationResult(StrictModel):
    """The natural filtration as a tuple of sigma algebras."""

    space: FiniteProbabilitySpace
    observations: tuple[tuple[str, ...], ...] = Field(default=())
    sigmas: tuple[FiniteSigmaAlgebra, ...] = Field(default=())


class DoobMartingaleResult(StrictModel):
    """The Doob martingale as canonical rational value vectors."""

    space: FiniteProbabilitySpace
    observations: tuple[tuple[str, ...], ...] = Field(default=())
    payoff: tuple[CanonicalRational, ...] = Field(min_length=1)
    martingale: tuple[FiniteRandomVariable, ...] = Field(default=())


__all__ = [
    "ConditionalExpectationRequest",
    "DoobMartingaleRequest",
    "DoobMartingaleResult",
    "FiltrationRequest",
    "FiltrationResult",
    "FromObservationRequest",
    "JoinRequest",
]
