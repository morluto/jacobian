"""Exact native finite delta-matroid construction."""

from __future__ import annotations

import json

from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidDistanceResult,
    DeltaMatroidRecognitionResult,
    require_twist_subset,
)
from jacobian.math.combinatorics.matroids.delta.extra import DeltaMatroidDirectSumResult
from jacobian.math.combinatorics.matroids.delta.values import (
    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS,
    MAX_DELTA_MEMBERSHIPS,
    DeltaMatroidAdmissionError,
    FiniteDeltaMatroid,
    first_symmetric_exchange_obstruction,
    require_delta_matroid_admission,
    require_delta_matroid_envelope,
    require_delta_matroid_exchange_work,
)

__all__ = [
    "direct_sum",
    "distance",
    "from_feasible_sets",
    "twist",
    "verify_from_feasible_sets",
    "width",
]


def direct_sum(
    left: FiniteDeltaMatroid, right: FiniteDeltaMatroid
) -> DeltaMatroidDirectSumResult:
    """Return the disjoint-ground direct sum, with concatenated ground axis."""
    from jacobian.math.combinatorics.matroids.delta.extra import (
        MAX_DIRECT_SUM_FEASIBLE_PAIRS,
        MAX_DIRECT_SUM_GROUND,
        MAX_DIRECT_SUM_OUTPUT_BYTES,
    )

    left_system = FiniteFeasibleSetSystem(ground=left.ground, feasible=left.feasible)
    right_system = FiniteFeasibleSetSystem(ground=right.ground, feasible=right.feasible)
    # Admit source envelopes and verify caller-authored values before composing.
    for source in (left_system, right_system):
        require_delta_matroid_envelope(source)
        require_delta_matroid_exchange_work(source)
        if first_symmetric_exchange_obstruction(source) is not None:
            raise ValueError("source feasible family is not a delta-matroid")

    ground_size = len(left.ground) + len(right.ground)
    if ground_size > MAX_DIRECT_SUM_GROUND:
        raise DeltaMatroidAdmissionError(
            "direct_sum_ground_exceeded",
            f"direct-sum ground exceeds {MAX_DIRECT_SUM_GROUND} elements",
        )
    if set(left.ground).intersection(right.ground):
        raise DeltaMatroidAdmissionError(
            "direct_sum_ground_overlap",
            "direct-sum ground labels must be disjoint",
        )
    pairs = len(left.feasible) * len(right.feasible)
    if pairs > MAX_DIRECT_SUM_FEASIBLE_PAIRS:
        raise DeltaMatroidAdmissionError(
            "direct_sum_work_exceeded",
            f"direct-sum feasible-pair work exceeds {MAX_DIRECT_SUM_FEASIBLE_PAIRS}",
        )
    membership_count = len(right.feasible) * sum(map(len, left.feasible)) + len(
        left.feasible
    ) * sum(map(len, right.feasible))
    if membership_count > MAX_DELTA_MEMBERSHIPS:
        raise DeltaMatroidAdmissionError(
            "memberships_exceeded",
            "direct-sum feasible memberships exceed the delta-matroid envelope",
        )
    # Every pair of output sets can differ in at most n elements. This sound
    # pre-product ceiling also ensures the result remains admissible by the
    # delta-matroid exchange checker.
    if pairs * pairs * ground_size * ground_size > MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS:
        raise DeltaMatroidAdmissionError(
            "candidate_work_exceeded",
            "direct-sum family exceeds the symmetric-exchange work envelope",
        )
    ground_json = json.dumps(left.ground + right.ground, ensure_ascii=False)
    digits = max(1, len(str(max(0, ground_size - 1))))
    family_bound = membership_count * (digits + 1) + 2 * pairs
    maps_bound = (len(left.ground) + len(right.ground)) * (digits + 2)
    total_bound = (
        len(left.model_dump_json().encode("utf-8"))
        + len(right.model_dump_json().encode("utf-8"))
        + len(ground_json.encode("utf-8"))
        + family_bound
        + maps_bound
        + 512
    )
    if total_bound > MAX_DIRECT_SUM_OUTPUT_BYTES:
        raise DeltaMatroidAdmissionError(
            "direct_sum_output_exceeded",
            f"direct-sum result exceeds the {MAX_DIRECT_SUM_OUTPUT_BYTES}-byte envelope",
        )

    rows = tuple(
        sorted(
            tuple(sorted((*a, *(len(left.ground) + i for i in b))))
            for a in left.feasible
            for b in right.feasible
        )
    )
    result = FiniteDeltaMatroid._from_kernel(
        FiniteFeasibleSetSystem(ground=left.ground + right.ground, feasible=rows)
    )
    return DeltaMatroidDirectSumResult.model_construct(
        left=left,
        right=right,
        direct_sum=result,
        left_injection=tuple(range(len(left.ground))),
        right_injection=tuple(range(len(left.ground), ground_size)),
    )


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


def distance(
    delta_matroid: FiniteDeltaMatroid, subset: tuple[int, ...]
) -> DeltaMatroidDistanceResult:
    """Return the exact symmetric-difference distance to feasibility.

    Ties are resolved by the canonical feasible-row order. The operation is
    linear in the retained family after source delta-matroid admission.
    """

    require_twist_subset(delta_matroid, subset)
    try:
        system = FiniteFeasibleSetSystem(
            ground=delta_matroid.ground, feasible=delta_matroid.feasible
        )
        require_delta_matroid_envelope(system)
    except DeltaMatroidAdmissionError:
        raise
    except (ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_valid",
            message=str(exc),
        ) from exc
    _require_delta_matroid_axiom(system)
    target = sum(1 << index for index in subset)
    nearest = min(
        delta_matroid.feasible,
        key=lambda row: (
            (target ^ sum(1 << index for index in row)).bit_count(),
            row,
        ),
    )
    value = (target ^ sum(1 << index for index in nearest)).bit_count()
    return DeltaMatroidDistanceResult._from_kernel(
        delta_matroid, subset, value, nearest
    )
