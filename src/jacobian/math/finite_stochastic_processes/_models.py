"""Typed wire contracts for finite stochastic process operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel
from jacobian.math.finite_stochastic_processes.values import (
    FiniteProbabilitySpace,
    FiniteRandomVariable,
    FiniteSigmaAlgebra,
)


class FromObservationRequest(StrictModel):
    """Construct a sigma algebra from an observation map."""

    space: FiniteProbabilitySpace
    observation: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_observation_matches_space(self) -> Self:
        if len(self.observation) != len(self.space.samples):
            raise ValueError("observation must have one entry per sample")
        return self


class JoinRequest(StrictModel):
    """Compute the join of two sigma algebras."""

    sigma1: FiniteSigmaAlgebra
    sigma2: FiniteSigmaAlgebra

    @model_validator(mode="after")
    def require_same_space(self) -> Self:
        if self.sigma1.space != self.sigma2.space:
            raise ValueError("sigma algebras must share the same probability space")
        return self


class ConditionalExpectationRequest(StrictModel):
    """Compute E[X | G]."""

    rv: FiniteRandomVariable
    sigma: FiniteSigmaAlgebra

    @model_validator(mode="after")
    def require_same_space(self) -> Self:
        if self.rv.space != self.sigma.space:
            raise ValueError(
                "random variable and sigma algebra must share the same probability space"
            )
        for value in self.rv.values:
            require_bounded_rational(
                value,
                max_digits=256,
                label="random-variable value",
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
                raise ValueError("observation must have one entry per sample")
        return self


class DoobMartingaleRequest(StrictModel):
    """Compute the Doob martingale of a payoff process."""

    space: FiniteProbabilitySpace
    observations: tuple[tuple[str, ...], ...] = Field(default=())
    payoff: tuple[CanonicalRational, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_payoff_matches_space(self) -> Self:
        if len(self.payoff) != len(self.space.samples):
            raise ValueError("payoff must have one entry per sample")
        for value in self.payoff:
            require_bounded_rational(
                value,
                max_digits=256,
                label="payoff",
            )
        for obs in self.observations:
            if len(obs) != len(self.space.samples):
                raise ValueError("observation must have one entry per sample")
        return self


class FiltrationResult(StrictModel):
    """The natural filtration as a tuple of sigma algebras."""

    sigmas: tuple[FiniteSigmaAlgebra, ...] = Field(default=())


class DoobMartingaleResult(StrictModel):
    """The Doob martingale as canonical rational value vectors."""

    martingale: tuple[tuple[CanonicalRational, ...], ...] = Field(default=())


__all__ = [
    "ConditionalExpectationRequest",
    "DoobMartingaleRequest",
    "DoobMartingaleResult",
    "FiltrationRequest",
    "FiltrationResult",
    "FromObservationRequest",
    "JoinRequest",
]
