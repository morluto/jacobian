"""Exact native finite delta-matroid construction."""

from __future__ import annotations

from typing import Literal

from pydantic import ValidationError

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidDistanceRequest,
    DeltaMatroidDistanceResult,
    DeltaMatroidRecognitionResult,
    require_twist_subset,
)
from jacobian.math.combinatorics.matroids.delta.values import (
    MAX_DELTA_DISTANCE_PROFILE_EVALUATIONS,
    MAX_DELTA_DISTANCE_PROFILE_STATES,
    MAX_DELTA_LABEL_BYTES,
    MAX_DELTA_MEMBERSHIPS,
    DeltaMatroidAdmissionError,
    DeltaMatroidDistanceProfile,
    FiniteDeltaMatroid,
    canonical_feasible_rows,
    first_symmetric_exchange_obstruction,
    require_delta_matroid_admission,
    require_delta_matroid_envelope,
    require_delta_matroid_exchange_work,
)
from jacobian.math.combinatorics.matroids.values import (
    MAX_FINITE_BASIS_COUNT,
    MAX_FINITE_BASIS_EXCHANGE_CHECKS,
    MAX_FINITE_BASIS_GROUND_SIZE,
    MAX_FINITE_BASIS_LABEL_BYTES,
    MAX_FINITE_BASIS_MEMBERSHIPS,
    MAX_FINITE_BASIS_TOTAL_LABEL_BYTES,
    FiniteBasisMatroid,
)

__all__ = [
    "distance_profile",
    "from_feasible_sets",
    "lower_matroid",
    "twist",
    "upper_matroid",
    "verify_from_feasible_sets",
    "width",
]


def _raise_direct_sum_admission(
    exc: DeltaMatroidAdmissionError, location: tuple[str, ...]
) -> NoReturn:
    error_type = (
        OperationResourceAdmissionError
        if exc.reason.endswith("_exceeded")
        else OperationDomainValidationError
    )
    raise error_type(
        location=location,
        code=f"delta_matroid.{exc.reason}",
        message=str(exc),
    ) from exc

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


def _preflight_extremal_source(delta_matroid: object) -> FiniteDeltaMatroid:
    """Validate native source shape and size before any structural copy."""
    if not isinstance(delta_matroid, FiniteDeltaMatroid):
        raise ValueError("source must be a FiniteDeltaMatroid value")
    try:
        ground = delta_matroid.ground
        feasible = delta_matroid.feasible
    except AttributeError as exc:
        raise ValueError("source is missing ground or feasible fields") from exc
    if not isinstance(ground, tuple) or not isinstance(feasible, tuple):
        raise ValueError("source ground and feasible family must be immutable tuples")
    if len(ground) > MAX_DELTA_MEMBERSHIPS + 1:
        raise DeltaMatroidAdmissionError(
            "ground_size_exceeded", "source ground axis exceeds conversion preflight"
        )
    if any(type(label) is not str for label in ground):
        raise ValueError("source ground labels must be exact strings")
    if len(feasible) > MAX_DELTA_MEMBERSHIPS + 1:
        raise DeltaMatroidAdmissionError(
            "row_count_exceeded", "source feasible-family row count exceeds preflight"
        )
    if not feasible:
        raise ValueError("a delta-matroid must have at least one feasible set")

    label_bytes = 0
    for label in ground:
        if not isinstance(label, str):
            raise ValueError("source ground labels must be strings")
        if len(label) > MAX_DELTA_LABEL_BYTES:
            raise DeltaMatroidAdmissionError(
                "label_bytes_exceeded", "source ground label exceeds its byte envelope"
            )
        try:
            label_bytes += len(label.encode("utf-8"))
        except UnicodeEncodeError:
            raise ValueError(
                "source ground labels must be UTF-8-representable"
            ) from None
        if label_bytes > MAX_DELTA_LABEL_BYTES:
            raise DeltaMatroidAdmissionError(
                "label_bytes_exceeded",
                "source ground labels exceed their byte envelope",
            )
    memberships = 0
    for row in feasible:
        memberships += _preflight_extremal_row(row, len(ground), memberships)
    return delta_matroid


def _preflight_extremal_row(
    row: object, ground_size: int, prior_memberships: int
) -> int:
    """Bound one forged feasible row before inspecting its indices."""
    if not isinstance(row, tuple):
        raise ValueError("source feasible rows must be immutable tuples")
    if len(row) > ground_size:
        raise DeltaMatroidAdmissionError(
            "row_size_exceeded", "source feasible row exceeds the ground axis"
        )
    if prior_memberships + len(row) > MAX_DELTA_MEMBERSHIPS:
        raise DeltaMatroidAdmissionError(
            "memberships_exceeded",
            "source feasible-family memberships exceed the envelope",
        )
    if any(type(index) is not int for index in row):
        raise ValueError("source feasible indices must be exact integers")
    return len(row)


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


def _extremal_matroid(
    delta_matroid: FiniteDeltaMatroid,
    *,
    extremum: Literal["minimum", "maximum"],
) -> FiniteBasisMatroid:
    """Derive and map the complete minimum- or maximum-size feasible family."""
    try:
        delta_matroid = _preflight_extremal_source(delta_matroid)
    except DeltaMatroidAdmissionError as exc:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code=f"delta_matroid.{exc.reason}",
            message=str(exc),
        ) from exc
    except (TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_valid",
            message=str(exc),
        ) from exc
    try:
        system = FiniteFeasibleSetSystem(
            ground=delta_matroid.ground, feasible=delta_matroid.feasible
        )
        if delta_matroid.feasible != canonical_feasible_rows(system):
            raise ValueError("delta-matroid feasible rows must be canonical")
        delta_matroid = FiniteDeltaMatroid._from_kernel(system)
    except (ValidationError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_valid",
            message=str(exc),
        ) from exc

    ground_size = len(system.ground)
    if ground_size > MAX_FINITE_BASIS_GROUND_SIZE:
        raise OperationResourceAdmissionError(
            location=("delta_matroid", "ground"),
            code="finite_basis_matroid.ground_size_bound",
            message=(
                "extremal matroid output requires at most "
                f"{MAX_FINITE_BASIS_GROUND_SIZE} ground labels"
            ),
        )
    label_bytes = 0
    for label in system.ground:
        encoded_length = len(label.encode("utf-8"))
        if encoded_length > MAX_FINITE_BASIS_LABEL_BYTES:
            raise OperationResourceAdmissionError(
                location=("delta_matroid", "ground"),
                code="finite_basis_matroid.ground_label_bound",
                message="a ground label exceeds the finite-basis per-label limit",
            )
        label_bytes += encoded_length
        if label_bytes > MAX_FINITE_BASIS_TOTAL_LABEL_BYTES:
            raise OperationResourceAdmissionError(
                location=("delta_matroid", "ground"),
                code="finite_basis_matroid.ground_label_total_bound",
                message="ground labels exceed the finite-basis total byte limit",
            )

    # Replay at this operation boundary: canonical structure alone does not
    # establish the source's symmetric-exchange axiom.
    try:
        require_delta_matroid_exchange_work(system)
    except DeltaMatroidAdmissionError as exc:
        if exc.reason == "labels_not_utf8":
            raise OperationDomainValidationError(
                location=("delta_matroid", "ground"),
                code="delta_matroid.source_not_valid",
                message=str(exc),
            ) from exc
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code=f"delta_matroid.{exc.reason}",
            message=str(exc),
        ) from exc
    obstruction = first_symmetric_exchange_obstruction(system)
    if obstruction is not None:
        raise OperationDomainValidationError(
            location=("delta_matroid",),
            code="delta_matroid.source_not_valid",
            message="source feasible family fails symmetric exchange",
        )

    sizes = tuple(len(row) for row in system.feasible)
    target_size = min(sizes) if extremum == "minimum" else max(sizes)
    basis_count = sum(row_size == target_size for row_size in sizes)
    if basis_count > MAX_FINITE_BASIS_COUNT:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code="finite_basis_matroid.basis_count_bound",
            message="extremal basis family exceeds the admitted row count",
        )
    memberships = basis_count * target_size
    if memberships > MAX_FINITE_BASIS_MEMBERSHIPS:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code="finite_basis_matroid.membership_bound",
            message="extremal basis family exceeds the admitted membership count",
        )
    exchange_bound = basis_count**2 * min(target_size, ground_size - target_size) ** 2
    if exchange_bound > MAX_FINITE_BASIS_EXCHANGE_CHECKS:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code="finite_basis_matroid.exchange_work_bound",
            message="extremal basis family exceeds the basis-exchange work limit",
        )

    # Materialize the selected canonical basis rows only after all output
    # bounds have passed.
    bases = tuple(
        row
        for row, row_size in zip(system.feasible, sizes, strict=True)
        if row_size == target_size
    )

    # The source exchange axiom and output envelope have already been checked;
    # the extremal-bases theorem establishes target basis exchange. Skip a
    # second quadratic exchange replay while binding this kernel-owned value.
    matroid = FiniteBasisMatroid._from_kernel(
        ground=system.ground,
        bases=bases,
    )
    return matroid


def lower_matroid(
    delta_matroid: FiniteDeltaMatroid,
) -> FiniteBasisMatroid:
    """Return the matroid whose bases are the minimum-size feasible sets."""
    return _extremal_matroid(delta_matroid, extremum="minimum")


def upper_matroid(
    delta_matroid: FiniteDeltaMatroid,
) -> FiniteBasisMatroid:
    """Return the matroid whose bases are the maximum-size feasible sets."""
    return _extremal_matroid(delta_matroid, extremum="maximum")


def distance_profile(
    delta_matroid: FiniteDeltaMatroid,
) -> DeltaMatroidDistanceProfile:
    """Return exact Hamming distance-to-feasibility data for every subset.

    A mask's bit ``i`` records whether ground index ``i`` is present. The
    profile is complete over all masks and counts every nearest feasible set.
    """
    request_checkpoint("before delta-matroid distance profile")

    try:
        if not isinstance(delta_matroid, FiniteDeltaMatroid):
            delta_matroid = FiniteDeltaMatroid.model_validate(delta_matroid)
        system = FiniteFeasibleSetSystem(
            ground=delta_matroid.ground, feasible=delta_matroid.feasible
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
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
    if not isinstance(value.ground, tuple) or not isinstance(value.feasible, tuple):
        raise OperationDomainValidationError(
            location=(name,),
            code="delta_matroid.source_not_valid",
            message=f"direct-sum {name} operand must use immutable canonical tuples",
        )
    if len(value.ground) > MAX_DELTA_MEMBERSHIPS + 1:
        raise DeltaMatroidAdmissionError(
            "ground_size_exceeded", "source ground axis exceeds the envelope"
        )
    if len(value.feasible) > MAX_DELTA_MEMBERSHIPS + 1:
        raise DeltaMatroidAdmissionError(
            "row_count_exceeded",
            "source feasible-family row count exceeds the envelope",
        )
    memberships = 0
    for row in value.feasible:
        if not isinstance(row, tuple) or len(row) > len(value.ground):
            raise OperationDomainValidationError(
                location=(name,),
                code="delta_matroid.source_not_valid",
                message=f"direct-sum {name} operand has a malformed feasible row",
            )
        memberships += len(row)
        if memberships > MAX_DELTA_MEMBERSHIPS:
            raise DeltaMatroidAdmissionError(
                "memberships_exceeded",
                "source feasible-family memberships exceed the envelope",
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
    DeltaMatroidDirectSumRequest,
    DeltaMatroidDirectSumResult,
        MAX_DIRECT_SUM_FEASIBLE_PAIRS,
        MAX_DIRECT_SUM_GROUND,
    )

    try:
        left = _admit_direct_sum_source(left, "left")
    except DeltaMatroidAdmissionError as exc:
        _raise_direct_sum_admission(exc, ("left",))
    try:
        right = _admit_direct_sum_source(right, "right")
    except DeltaMatroidAdmissionError as exc:
        _raise_direct_sum_admission(exc, ("right",))
    left_system = FiniteFeasibleSetSystem(ground=left.ground, feasible=left.feasible)
    right_system = FiniteFeasibleSetSystem(ground=right.ground, feasible=right.feasible)
    # Admit source envelopes and verify caller-authored values before composing.
    for name, source in (("left", left_system), ("right", right_system)):
        try:
            require_delta_matroid_envelope(source)
            require_delta_matroid_exchange_work(source)
            if first_symmetric_exchange_obstruction(source) is not None:
                raise OperationDomainValidationError(
                    location=(name,),
                    code="delta_matroid.source_not_valid",
                    message="direct-sum source is not a delta-matroid",
                )
        except DeltaMatroidAdmissionError as exc:
            _raise_direct_sum_admission(exc, (name,))

    ground_size = len(left.ground) + len(right.ground)
    if ground_size > MAX_DIRECT_SUM_GROUND:
        _raise_direct_sum_admission(
            DeltaMatroidAdmissionError(
                "direct_sum_ground_exceeded",
                f"direct-sum ground exceeds {MAX_DIRECT_SUM_GROUND} elements",
            ),
            ("left", "right"),
        )
    if set(left.ground).intersection(right.ground):
        _raise_direct_sum_admission(
            DeltaMatroidAdmissionError(
                "direct_sum_ground_overlap",
                "direct-sum ground labels must be disjoint",
            ),
            ("left", "right", "ground"),
        )
    pairs = len(left.feasible) * len(right.feasible)
    if pairs > MAX_DIRECT_SUM_FEASIBLE_PAIRS:
        _raise_direct_sum_admission(
            DeltaMatroidAdmissionError(
                "direct_sum_work_exceeded",
                f"direct-sum feasible-pair work exceeds {MAX_DIRECT_SUM_FEASIBLE_PAIRS}",
            ),
            ("left", "right"),
        )
    membership_count = len(right.feasible) * sum(map(len, left.feasible)) + len(
        left.feasible
    ) * sum(map(len, right.feasible))
    if membership_count > MAX_DELTA_MEMBERSHIPS:
        _raise_direct_sum_admission(
            DeltaMatroidAdmissionError(
                "memberships_exceeded",
                "direct-sum feasible memberships exceed the delta-matroid envelope",
            ),
            ("left", "right"),
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
        _raise_direct_sum_admission(
            DeltaMatroidAdmissionError(
                "label_bytes_exceeded",
                "direct-sum ground labels exceed the delta-matroid label envelope",
            ),
            ("left", "right", "ground"),
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


def distance(
    delta_matroid: FiniteDeltaMatroid, subset: tuple[int, ...]
) -> DeltaMatroidDistanceResult:
    """Return the exact symmetric-difference distance to feasibility.

    Ties are resolved by the canonical feasible-row order. The operation is
    linear in the retained family after source delta-matroid admission.
    """

    delta_matroid = _admit_direct_sum_source(delta_matroid, "delta_matroid")
    try:
        require_twist_subset(delta_matroid, subset)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("subset",),
            code="delta_matroid.twist_subset",
            message="subset must be a canonical in-range ground-index tuple",
        ) from exc
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
