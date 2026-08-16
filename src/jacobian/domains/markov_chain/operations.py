"""Domain adapter for Markov chain operations."""

from __future__ import annotations

from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.markov_chain import (
    ErgodicDecisionResult,
    StationaryDistributionResult,
    TransitionMatrixRequest,
)
from jacobian.math.markov_chain import is_ergodic, stationary_distribution


def compute_stationary_distribution(
    request: TransitionMatrixRequest,
) -> StationaryDistributionResult:
    matrix = [[{"num": c.num, "den": c.den} for c in row] for row in request.matrix]
    dist = stationary_distribution(matrix)
    return StationaryDistributionResult(
        distribution=tuple(
            CanonicalRational.from_integer_ratio(int(v.p), int(v.q)) for v in dist
        )
    )


def compute_ergodic_decision(request: TransitionMatrixRequest) -> ErgodicDecisionResult:
    matrix = [[{"num": c.num, "den": c.den} for c in row] for row in request.matrix]
    ergodic = is_ergodic(matrix)
    return ErgodicDecisionResult(
        is_ergodic=ergodic,
        is_irreducible=ergodic,
        is_aperdiodic=ergodic,
    )
