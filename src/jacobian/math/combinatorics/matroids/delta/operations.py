"""Exact native finite delta-matroid construction."""

from __future__ import annotations

from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidRecognitionResult,
)
from jacobian.math.combinatorics.matroids.delta.values import (
    FiniteDeltaMatroid,
    first_symmetric_exchange_obstruction,
    require_delta_matroid_admission,
)

__all__ = ["from_feasible_sets", "verify_from_feasible_sets"]


def from_feasible_sets(
    system: FiniteFeasibleSetSystem,
) -> DeltaMatroidRecognitionResult:
    """Recognize one complete feasible family by exhaustive symmetric exchange."""

    require_delta_matroid_admission(system)
    obstruction = first_symmetric_exchange_obstruction(system)
    if obstruction is not None:
        return DeltaMatroidRecognitionResult._from_kernel(
            system,
            obstruction=obstruction,
        )
    return DeltaMatroidRecognitionResult._from_kernel(
        system,
        delta_matroid=FiniteDeltaMatroid._from_kernel(system),
    )


def verify_from_feasible_sets(claim: DeltaMatroidRecognitionResult) -> bool:
    """Return whether a recognition claim matches its retained feasible family."""
    try:
        require_delta_matroid_admission(claim.source)
    except ValueError:
        return False
    obstruction = first_symmetric_exchange_obstruction(claim.source)
    if obstruction is not None:
        return (
            claim.status == "NOT_A_DELTA_MATROID"
            and claim.delta_matroid is None
            and claim.obstruction == obstruction
        )
    return (
        claim.status == "DELTA_MATROID"
        and claim.obstruction is None
        and claim.delta_matroid == FiniteDeltaMatroid._from_kernel(claim.source)
    )
