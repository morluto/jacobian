"""Domain adapter for finite stochastic process operations."""

from __future__ import annotations

from jacobian.math.probability.stochastic_processes._models import (
    ConditionalExpectationRequest,
    DoobMartingaleRequest,
    DoobMartingaleResult,
    FiltrationRequest,
    FiltrationResult,
    FromObservationRequest,
    JoinRequest,
)
from jacobian.math.probability.stochastic_processes.operations import (
    conditional_expectation,
    doob_martingale,
    filtration_natural,
    sigma_algebra_from_observation,
)
from jacobian.math.probability.stochastic_processes.values import (
    FiniteRandomVariable,
    FiniteSigmaAlgebra,
)

__all__ = [
    "compute_conditional_expectation",
    "compute_doob_martingale",
    "compute_filtration",
    "compute_join",
    "compute_sigma_from_observation",
]


def compute_sigma_from_observation(
    request: FromObservationRequest,
) -> FiniteSigmaAlgebra:
    return sigma_algebra_from_observation(request.space, request.observation)


def compute_join(request: JoinRequest) -> FiniteSigmaAlgebra:
    from jacobian.math.probability.stochastic_processes.operations import (
        sigma_algebra_join,
    )

    return sigma_algebra_join(request.sigma1, request.sigma2)


def compute_conditional_expectation(
    request: ConditionalExpectationRequest,
) -> FiniteRandomVariable:
    return conditional_expectation(request.rv, request.sigma)


def compute_filtration(request: FiltrationRequest) -> FiltrationResult:
    sigmas = filtration_natural(request.space, request.observations)
    return FiltrationResult(sigmas=sigmas)


def compute_doob_martingale(
    request: DoobMartingaleRequest,
) -> DoobMartingaleResult:
    result = doob_martingale(request.space, request.observations, request.payoff)
    return DoobMartingaleResult(martingale=result)
