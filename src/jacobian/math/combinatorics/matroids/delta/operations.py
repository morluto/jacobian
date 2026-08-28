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

__all__ = ["from_feasible_sets"]


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
