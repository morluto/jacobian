"""Domain adapter for Markov chain operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.markov_chain import (
    ErgodicDecisionResult,
    MixingTimeRequest,
    MixingTimeResult,
    StationaryDistributionResult,
    TransitionMatrixRequest,
)
from jacobian.math.markov_chain import (
    ergodic_properties,
    mixing_time,
    stationary_distribution,
)


def compute_stationary_distribution(
    request: TransitionMatrixRequest,
) -> StationaryDistributionResult:
    matrix = [[{"num": c.num, "den": c.den} for c in row] for row in request.matrix]
    dist = stationary_distribution(matrix)  # type: ignore[no-untyped-call]
    return StationaryDistributionResult(
        distribution=tuple(
            CanonicalRational.from_integer_ratio(int(v.p), int(v.q)) for v in dist
        )
    )


def compute_ergodic_decision(request: TransitionMatrixRequest) -> ErgodicDecisionResult:
    matrix = [[{"num": c.num, "den": c.den} for c in row] for row in request.matrix]
    irreducible, aperiodic = ergodic_properties(matrix)  # type: ignore[no-untyped-call]
    return ErgodicDecisionResult(
        is_ergodic=irreducible and aperiodic,
        is_irreducible=irreducible,
        is_aperiodic=aperiodic,
    )


def compute_mixing_time(request: MixingTimeRequest) -> MixingTimeResult:
    matrix = [[{"num": c.num, "den": c.den} for c in row] for row in request.matrix]
    irreducible, aperiodic = ergodic_properties(matrix)  # type: ignore[no-untyped-call]
    if not (irreducible and aperiodic):
        return MixingTimeResult(
            status="NOT_ERGODIC",
            epsilon=request.epsilon,
            max_steps=request.max_steps,
            steps_examined=0,
        )
    stationary = stationary_distribution(matrix)  # type: ignore[no-untyped-call]
    stationary_fractions = tuple(
        Fraction(int(value.p), int(value.q)) for value in stationary
    )
    result = mixing_time(
        tuple(tuple(value.as_fraction() for value in row) for row in request.matrix),
        stationary_fractions,
        request.epsilon.as_fraction(),
        request.max_steps,
    )
    return MixingTimeResult(
        status="FOUND" if result.mixing_time is not None else "BOUND_EXCEEDED",
        epsilon=request.epsilon,
        max_steps=request.max_steps,
        steps_examined=result.steps_examined,
        mixing_time=result.mixing_time,
        max_total_variation_distance=CanonicalRational.from_fraction(
            result.max_total_variation_distance
        ),
    )
