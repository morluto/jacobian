"""Exact native finite delta-matroid construction."""

from __future__ import annotations

from typing import Literal

from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids.delta._models import (
    DeltaMatroidExtremalMatroidResult,
    DeltaMatroidRecognitionResult,
    require_twist_subset,
)
from jacobian.math.combinatorics.matroids.delta.values import (
    MAX_DELTA_LABEL_BYTES,
    MAX_DELTA_MEMBERSHIPS,
    DeltaMatroidAdmissionError,
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
    "from_feasible_sets",
    "lower_matroid",
    "twist",
    "upper_matroid",
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


def _preflight_extremal_source(delta_matroid: object) -> FiniteDeltaMatroid:
    """Validate native source shape and size before any structural copy."""
    if not isinstance(delta_matroid, FiniteDeltaMatroid):
        raise ValueError("source must be a FiniteDeltaMatroid value")
    ground = delta_matroid.ground
    feasible = delta_matroid.feasible
    if not isinstance(ground, tuple) or not isinstance(feasible, tuple):
        raise ValueError("source ground and feasible family must be immutable tuples")
    if len(ground) > MAX_DELTA_MEMBERSHIPS + 1:
        raise DeltaMatroidAdmissionError(
            "ground_size_exceeded", "source ground axis exceeds conversion preflight"
        )
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
        if not isinstance(row, tuple):
            raise ValueError("source feasible rows must be immutable tuples")
        memberships += len(row)
        if memberships > MAX_DELTA_MEMBERSHIPS:
            raise DeltaMatroidAdmissionError(
                "memberships_exceeded",
                "source feasible-family memberships exceed the envelope",
            )
    return delta_matroid


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
) -> DeltaMatroidExtremalMatroidResult:
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

    # Materialize the selected basis rows and their identity map only after all
    # output bounds have passed.
    source_indices = tuple(
        index for index, row_size in enumerate(sizes) if row_size == target_size
    )
    bases = tuple(system.feasible[index] for index in source_indices)

    # The source exchange axiom and output envelope have already been checked;
    # the extremal-bases theorem establishes target basis exchange. Skip a
    # second quadratic exchange replay while binding this kernel-owned value.
    matroid = FiniteBasisMatroid._from_kernel(
        ground=system.ground,
        bases=bases,
    )
    return DeltaMatroidExtremalMatroidResult._from_kernel(
        delta_matroid,
        matroid,
        source_indices,
        extremum,
    )


def lower_matroid(
    delta_matroid: FiniteDeltaMatroid,
) -> DeltaMatroidExtremalMatroidResult:
    """Return the matroid whose bases are the minimum-size feasible sets."""
    return _extremal_matroid(delta_matroid, extremum="minimum")


def upper_matroid(
    delta_matroid: FiniteDeltaMatroid,
) -> DeltaMatroidExtremalMatroidResult:
    """Return the matroid whose bases are the maximum-size feasible sets."""
    return _extremal_matroid(delta_matroid, extremum="maximum")
