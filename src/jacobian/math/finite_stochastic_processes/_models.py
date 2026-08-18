"""Typed wire contracts for finite stochastic process operations."""

from __future__ import annotations

from pydantic import Field

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


class JoinRequest(StrictModel):
    """Compute the join of two sigma algebras."""

    sigma1: FiniteSigmaAlgebra
    sigma2: FiniteSigmaAlgebra


class ConditionalExpectationRequest(StrictModel):
    """Compute E[X | G]."""

    rv: FiniteRandomVariable
    sigma: FiniteSigmaAlgebra


class FiltrationRequest(StrictModel):
    """Compute the natural filtration of observations."""

    space: FiniteProbabilitySpace
    observations: tuple[tuple[str, ...], ...] = Field(default=())


class DoobMartingaleRequest(StrictModel):
    """Compute the Doob martingale of a payoff process."""

    space: FiniteProbabilitySpace
    observations: tuple[tuple[str, ...], ...] = Field(default=())
    payoff: tuple[str, ...] = Field(min_length=1)


class FiltrationResult(StrictModel):
    """The natural filtration as a tuple of sigma algebra dicts."""

    sigmas: tuple[dict[str, object], ...] = Field(default=())


class DoobMartingaleResult(StrictModel):
    """The Doob martingale as a tuple of rational-string value tuples."""

    martingale: tuple[tuple[str, ...], ...] = Field(default=())


__all__ = [
    "ConditionalExpectationRequest",
    "DoobMartingaleRequest",
    "DoobMartingaleResult",
    "FiltrationRequest",
    "FiltrationResult",
    "FromObservationRequest",
    "JoinRequest",
]
