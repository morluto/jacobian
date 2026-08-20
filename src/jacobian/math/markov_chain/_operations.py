"""Domain-owned Markov chain operations."""

from __future__ import annotations

from jacobian._exact import CanonicalRational
from jacobian.math.markov_chain import (
    ergodic_properties,
    mixing_time,
)
from jacobian.math.markov_chain._models import (
    ErgodicDecisionResult,
    ExtremeStationaryDistribution,
    MixingTimeRequest,
    MixingTimeResult,
    StationaryDistributionRequest,
    StationaryDistributionResult,
    TransitionMatrixRequest,
)
from jacobian.math.markov_chain.operations import _stationary_distribution_extremes


def compute_mixing_time(request: MixingTimeRequest) -> MixingTimeResult:
    matrix = tuple(
        tuple(value.as_fraction() for value in row) for row in request.matrix
    )
    irreducible, aperiodic = ergodic_properties(request)
    if not (irreducible and aperiodic):
        return MixingTimeResult(
            status="NOT_ERGODIC",
            epsilon=request.epsilon,
            max_steps=request.max_steps,
            steps_examined=0,
        )
    extremes = _stationary_distribution_extremes(request)
    stationary = extremes[0][1]
    outcome = mixing_time(
        matrix, stationary, request.epsilon.as_fraction(), request.max_steps
    )
    distance = CanonicalRational.from_integer_ratio(
        outcome.max_total_variation_distance.numerator,
        outcome.max_total_variation_distance.denominator,
    )
    return MixingTimeResult(
        status="FOUND" if outcome.mixing_time is not None else "BOUND_EXCEEDED",
        epsilon=request.epsilon,
        max_steps=request.max_steps,
        steps_examined=outcome.steps_examined,
        mixing_time=outcome.mixing_time,
        max_total_variation_distance=distance,
    )


def compute_stationary_distribution(
    request: StationaryDistributionRequest,
) -> StationaryDistributionResult:
    extremes = _stationary_distribution_extremes(request)
    return StationaryDistributionResult(
        extreme_distributions=tuple(
            ExtremeStationaryDistribution(
                closed_class=closed_class,
                distribution=tuple(
                    CanonicalRational.from_integer_ratio(
                        value.numerator, value.denominator
                    )
                    for value in distribution
                ),
            )
            for closed_class, distribution in extremes
        ),
        unique=len(extremes) == 1,
    )


def compute_ergodic_decision(request: TransitionMatrixRequest) -> ErgodicDecisionResult:
    irreducible, aperiodic = ergodic_properties(request)
    return ErgodicDecisionResult(
        is_ergodic=irreducible and aperiodic,
        is_irreducible=irreducible,
        is_aperiodic=aperiodic,
    )
