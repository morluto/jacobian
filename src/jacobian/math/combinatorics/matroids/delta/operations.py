"""Exact native finite delta-matroid construction."""

from __future__ import annotations

from pydantic import ValidationError

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidRecognitionResult,
    require_twist_subset,
)
from jacobian.math.combinatorics.matroids.delta.values import (
    MAX_DELTA_DISTANCE_PROFILE_EVALUATIONS,
    MAX_DELTA_DISTANCE_PROFILE_STATES,
    MAX_DELTA_MEMBERSHIPS,
    DeltaMatroidAdmissionError,
    DeltaMatroidDistanceProfile,
    FiniteDeltaMatroid,
    first_symmetric_exchange_obstruction,
    require_delta_matroid_admission,
    require_delta_matroid_envelope,
    require_delta_matroid_exchange_work,
)

__all__ = [
    "distance_profile",
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


def _require_delta_matroid_axiom(system: FiniteFeasibleSetSystem) -> None:
    """Replay candidate-work and exchange after the source envelope is known."""

    require_delta_matroid_exchange_work(system)
    obstruction = first_symmetric_exchange_obstruction(system)
    if obstruction is not None:
        raise ValueError("source feasible family is not a delta-matroid")


def twist(
    delta_matroid: FiniteDeltaMatroid, subset: tuple[int, ...]
) -> FiniteDeltaMatroid:
    """Return the delta-matroid twist by ``subset``.

    Feasible sets are transformed as ``F △ X``.  The source axiom is replayed
    at this consumer boundary because a deserialized ``FiniteDeltaMatroid`` is
    caller-authored rather than trusted producer output.
    """

    require_twist_subset(delta_matroid, subset)
    system = FiniteFeasibleSetSystem(
        ground=delta_matroid.ground, feasible=delta_matroid.feasible
    )
    require_delta_matroid_envelope(system)
    twist_subset = frozenset(subset)
    projected_memberships = sum(
        len(row) + len(twist_subset) - 2 * len(twist_subset.intersection(row))
        for row in delta_matroid.feasible
    )
    if projected_memberships > MAX_DELTA_MEMBERSHIPS:
        raise DeltaMatroidAdmissionError(
            "memberships_exceeded",
            "twisted feasible-family memberships exceed the output envelope",
        )
    _require_delta_matroid_axiom(system)
    rows = tuple(
        sorted(
            tuple(sorted(frozenset(row) ^ twist_subset))
            for row in delta_matroid.feasible
        )
    )
    twisted_system = FiniteFeasibleSetSystem(
        ground=delta_matroid.ground,
        feasible=rows,
    )
    # Twisting preserves symmetric differences, hence symmetric exchange and
    # its candidate-work bound. The output membership bound was admitted above.
    return FiniteDeltaMatroid._from_kernel(twisted_system)


def width(delta_matroid: FiniteDeltaMatroid) -> int:
    """Return the delta-matroid width ``max |F| - min |F|``.

    The width is a linear projection of the feasible rows, so recognition
    admission is not replayed; the structural carrier is reconstructed so a
    forged instance with malformed rows cannot produce an incorrect value.
    """

    try:
        system = FiniteFeasibleSetSystem(
            ground=delta_matroid.ground, feasible=delta_matroid.feasible
        )
        if not system.feasible:
            raise ValueError("a delta-matroid has at least one feasible set")
    except (ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_valid",
            message=str(exc),
        ) from exc
    sizes = tuple(len(row) for row in system.feasible)
    return max(sizes) - min(sizes)


def distance_profile(
    delta_matroid: FiniteDeltaMatroid,
) -> DeltaMatroidDistanceProfile:
    """Return exact Hamming distance-to-feasibility data for every subset.

    A mask's bit ``i`` records whether ground index ``i`` is present. The
    profile is complete over all masks and counts every nearest feasible set.
    """

    try:
        system = FiniteFeasibleSetSystem(
            ground=delta_matroid.ground, feasible=delta_matroid.feasible
        )
    except (ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_valid",
            message=str(exc),
        ) from exc

    ground_size = len(system.ground)
    if ground_size > MAX_DELTA_DISTANCE_PROFILE_STATES.bit_length() - 1:
        raise DeltaMatroidAdmissionError(
            "distance_profile_states_exceeded",
            "the complete ground-subset profile exceeds the subset-state envelope",
        )
    state_count = 1 << ground_size
    if state_count > MAX_DELTA_DISTANCE_PROFILE_STATES:
        raise DeltaMatroidAdmissionError(
            "distance_profile_states_exceeded",
            "the complete ground-subset profile exceeds the subset-state envelope",
        )
    distance_evaluations = state_count * len(system.feasible)
    if distance_evaluations > MAX_DELTA_DISTANCE_PROFILE_EVALUATIONS:
        raise DeltaMatroidAdmissionError(
            "distance_profile_work_exceeded",
            "subset/feasible-set distance work exceeds its admitted envelope",
        )

    require_delta_matroid_admission(system)
    obstruction = first_symmetric_exchange_obstruction(system)
    if obstruction is not None:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_valid",
            message="source feasible family fails symmetric exchange",
        )

    feasible_masks = tuple(sum(1 << index for index in row) for row in system.feasible)
    distances: list[int] = []
    nearest_counts: list[int] = []
    histogram = [0] * (ground_size + 1)
    evaluations = 0
    for subset_mask in range(state_count):
        minimum = ground_size + 1
        nearest_count = 0
        for feasible_mask in feasible_masks:
            evaluations += 1
            if evaluations % 4_096 == 0:
                request_checkpoint("during delta-matroid distance profile")
            distance = (subset_mask ^ feasible_mask).bit_count()
            if distance < minimum:
                minimum = distance
                nearest_count = 1
            elif distance == minimum:
                nearest_count += 1
        distances.append(minimum)
        nearest_counts.append(nearest_count)
        histogram[minimum] += 1

    return DeltaMatroidDistanceProfile._from_kernel(
        FiniteDeltaMatroid._from_kernel(system),
        tuple(distances),
        tuple(nearest_counts),
        tuple(histogram),
    )
