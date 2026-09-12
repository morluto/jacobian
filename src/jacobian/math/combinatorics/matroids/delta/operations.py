"""Exact native finite delta-matroid construction."""

from __future__ import annotations

from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidRecognitionResult,
    DeltaMatroidTwistRequest,
    DeltaMatroidTwistResult,
    DeltaMatroidWidthRequest,
    DeltaMatroidWidthResult,
)
from jacobian.math.combinatorics.matroids.delta.values import (
    MAX_DELTA_MEMBERSHIPS,
    DeltaMatroidAdmissionError,
    FiniteDeltaMatroid,
    first_symmetric_exchange_obstruction,
    require_delta_matroid_admission,
)

__all__ = [
    "from_feasible_sets",
    "twist",
    "verify_from_feasible_sets",
    "width",
]


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


def _require_delta_matroid(value: FiniteDeltaMatroid) -> None:
    """Re-establish the source axiom before consuming a serialized claim."""

    system = FiniteFeasibleSetSystem(ground=value.ground, feasible=value.feasible)
    require_delta_matroid_admission(system)
    obstruction = first_symmetric_exchange_obstruction(system)
    if obstruction is not None:
        raise ValueError("source feasible family is not a delta-matroid")


def twist(request: DeltaMatroidTwistRequest) -> DeltaMatroidTwistResult:
    """Return the delta-matroid twist by ``subset``.

    Feasible sets are transformed as ``F △ X``.  The source axiom is replayed
    at this consumer boundary because a deserialized ``FiniteDeltaMatroid`` is
    caller-authored rather than trusted producer output.
    """

    _require_delta_matroid(request.delta_matroid)
    subset = frozenset(request.subset)
    projected_memberships = sum(
        len(row) + len(subset) - 2 * len(subset.intersection(row))
        for row in request.delta_matroid.feasible
    )
    if projected_memberships > MAX_DELTA_MEMBERSHIPS:
        raise DeltaMatroidAdmissionError(
            "memberships_exceeded",
            "twisted feasible-family memberships exceed the output envelope",
        )
    rows = tuple(
        sorted(
            tuple(sorted(frozenset(row) ^ subset))
            for row in request.delta_matroid.feasible
        )
    )
    twisted_system = FiniteFeasibleSetSystem(
        ground=request.delta_matroid.ground,
        feasible=rows,
    )
    # Twisting preserves symmetric differences, hence symmetric exchange and
    # its candidate-work bound. The output membership bound was admitted above.
    return DeltaMatroidTwistResult._from_kernel(
        request,
        FiniteDeltaMatroid._from_kernel(twisted_system),
    )


def width(request: DeltaMatroidWidthRequest) -> DeltaMatroidWidthResult:
    """Return the delta-matroid width ``max |F| - min |F|``."""

    _require_delta_matroid(request.delta_matroid)
    sizes = tuple(len(row) for row in request.delta_matroid.feasible)
    return DeltaMatroidWidthResult._from_kernel(
        request,
        max(sizes) - min(sizes),
    )
