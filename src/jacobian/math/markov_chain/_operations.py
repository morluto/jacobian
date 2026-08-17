"""Domain-owned Markov chain operations."""

from __future__ import annotations

from jacobian._exact import CanonicalRational
from jacobian.math.markov_chain import ergodic_properties, stationary_distribution
from jacobian.math.markov_chain._models import (
    ErgodicDecisionResult,
    StationaryDistributionResult,
    TransitionMatrixRequest,
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
