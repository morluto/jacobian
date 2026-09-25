"""Exact native finite delta-matroid construction."""

from __future__ import annotations

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
    MAX_DELTA_MEMBERSHIPS,
    DeltaMatroidAdmissionError,
    FiniteDeltaMatroid,
    first_symmetric_exchange_obstruction,
    require_delta_matroid_admission,
    require_delta_matroid_envelope,
    require_delta_matroid_exchange_work,
)


def _admit_direct_sum_source(value: object, name: str) -> FiniteDeltaMatroid:
    """Revalidate one native direct-sum operand before any field access.

    A deserialized or ``model_construct``-forged operand is caller-authored, so
    the carrier is reconstructed through the canonical contract and malformed
    values map to the operation's declared domain error instead of leaking
    attribute, indexing, or Pydantic exceptions.
    """

    if type(value) is not FiniteDeltaMatroid:
        raise OperationDomainValidationError(
            location=(name,),
            code="delta_matroid.source_not_valid",
            message=f"direct-sum {name} operand must be a finite delta-matroid",
        )
    try:
        return FiniteDeltaMatroid.model_validate(value.model_dump(mode="python"))
    except Exception as exc:
        raise OperationDomainValidationError(
            location=(name,),
            code="delta_matroid.source_not_valid",
            message=f"direct-sum {name} operand is not a canonical finite "
            "delta-matroid",
        ) from exc


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
    """Return the disjoint-ground direct sum, with concatenated ground axis.

    Both operands are revalidated as canonical carriers before any field is
    read.  The pairwise-union family satisfies symmetric exchange by the
    direct-sum theorem, so admission bounds the actual product construction
    and retained output through the ground, feasible-pair, membership, and
    label envelopes rather than replaying the recognition axiom.
    """
    from jacobian.math.combinatorics.matroids.delta.extra import (
        MAX_DIRECT_SUM_FEASIBLE_PAIRS,
        MAX_DIRECT_SUM_GROUND,
    )

    left = _admit_direct_sum_source(left, "left")
    right = _admit_direct_sum_source(right, "right")
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
    try:
        combined_label_bytes = sum(
            len(label.encode("utf-8")) for label in (*left.ground, *right.ground)
        )
    except UnicodeEncodeError as exc:
        raise OperationDomainValidationError(
            location=("ground",),
            code="delta_matroid.labels_not_utf8",
            message="direct-sum labels must be UTF-8-representable",
        ) from exc
    from jacobian.math.combinatorics.matroids.delta.values import MAX_DELTA_LABEL_BYTES

    if combined_label_bytes > MAX_DELTA_LABEL_BYTES:
        raise DeltaMatroidAdmissionError(
            "label_bytes_exceeded",
            "direct-sum ground labels exceed the delta-matroid label envelope",
        )
    # The pair and membership bounds above exactly bound the result family's
    # rows and index positions, so the direct sum materializes within the
    # carrier's declared cardinality envelopes without estimating encoded
    # bytes and without charging a symmetric-exchange replay this operation
    # never performs.

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

    delta_matroid = _admit_direct_sum_source(delta_matroid, "delta_matroid")
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

    delta_matroid = _admit_direct_sum_source(delta_matroid, "delta_matroid")
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
