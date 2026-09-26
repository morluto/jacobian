"""Typed wire contracts for finite delta-matroid operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids.delta.values import (
    MAX_DELTA_DISTANCE_PROFILE_EVALUATIONS,
    MAX_DELTA_DISTANCE_PROFILE_STATES,
    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS,
    MAX_DELTA_LABEL_BYTES,
    MAX_DELTA_MEMBERSHIPS,
    DeltaMatroidObstruction,
    FiniteDeltaMatroid,
)
from jacobian.math.combinatorics.matroids.values import (
    MAX_FINITE_BASIS_COUNT,
    MAX_FINITE_BASIS_EXCHANGE_CHECKS,
    MAX_FINITE_BASIS_GROUND_SIZE,
    MAX_FINITE_BASIS_LABEL_BYTES,
    MAX_FINITE_BASIS_MEMBERSHIPS,
    MAX_FINITE_BASIS_TOTAL_LABEL_BYTES,
)

MAX_DELTA_EXTREMAL_SOURCE_GROUND_LABELS = MAX_DELTA_MEMBERSHIPS + 1


def require_twist_subset(
    delta_matroid: FiniteDeltaMatroid, subset: tuple[int, ...]
) -> None:
    """Validate the canonical twist axis for native and wire callers."""

    if any(type(index) is not int for index in subset):
        raise _validation_error("subset_index_type", "twist indices must be integers")
    if subset != tuple(sorted(set(subset))):
        raise _validation_error(
            "subset_not_canonical", "twist subset must be sorted and distinct"
        )
    if any(index < 0 or index >= len(delta_matroid.ground) for index in subset):
        raise _validation_error(
            "subset_out_of_range", "twist subset index is outside the ground set"
        )


class DeltaMatroidTwistRequest(StrictModel):
    """Twist a delta-matroid by a canonical ground-index subset."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Twist one complete finite delta-matroid by a sorted ground-index "
                "subset. Source recognition and the twisted family share the "
                f"{MAX_DELTA_MEMBERSHIPS}-membership, {MAX_DELTA_LABEL_BYTES}-byte "
                "label, and "
                f"{MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS}-candidate envelopes."
            ),
            "admission_limits": {
                "max_feasible_set_memberships": MAX_DELTA_MEMBERSHIPS,
                "max_ground_label_utf8_bytes": MAX_DELTA_LABEL_BYTES,
                "max_symmetric_exchange_candidate_checks_per_replay": (
                    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS
                ),
            },
        }
    )

    delta_matroid: FiniteDeltaMatroid = Field(
        description=(
            "Canonical finite delta-matroid. Twist admission allows at most "
            f"{MAX_DELTA_MEMBERSHIPS} total feasible-row memberships, "
            f"{MAX_DELTA_LABEL_BYTES} UTF-8 ground-label bytes, and "
            f"{MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS} symmetric-exchange candidate "
            "checks before using the source family."
        )
    )
    subset: tuple[int, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical_subset(self) -> Self:
        require_twist_subset(self.delta_matroid, self.subset)
        return self


class DeltaMatroidTwistResult(DeltaMatroidTwistRequest):
    """The exact feasible family obtained by symmetric-difference twist."""

    twisted: FiniteDeltaMatroid

    @classmethod
    def _from_kernel(
        cls,
        delta_matroid: FiniteDeltaMatroid,
        subset: tuple[int, ...],
        twisted: FiniteDeltaMatroid,
    ) -> Self:
        return cls.model_construct(
            delta_matroid=delta_matroid,
            subset=subset,
            twisted=twisted,
        )


class DeltaMatroidWidthRequest(StrictModel):
    """Compute the width (largest minus smallest feasible-set size)."""

    delta_matroid: FiniteDeltaMatroid


class DeltaMatroidWidthResult(DeltaMatroidWidthRequest):
    width: int = Field(ge=0)

    @classmethod
    def _from_kernel(cls, delta_matroid: FiniteDeltaMatroid, width: int) -> Self:
        return cls.model_construct(delta_matroid=delta_matroid, width=width)


class DeltaMatroidDistanceProfileRequest(StrictModel):
    """Compute distance to feasibility for every subset of the ground set."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute the Hamming distance from every ground subset to the "
                "complete feasible family, plus nearest-set counts and a "
                "distance histogram. Ground subsets are ordered by their "
                "integer bit mask, where bit i denotes ground index i. The "
                "profile admits at most "
                f"{MAX_DELTA_DISTANCE_PROFILE_STATES} subset states and "
                f"{MAX_DELTA_DISTANCE_PROFILE_EVALUATIONS} subset/feasible-row "
                "distance evaluations."
            ),
            "admission_limits": {
                "max_subset_states": MAX_DELTA_DISTANCE_PROFILE_STATES,
                "max_subset_feasible_row_evaluations": (
                    MAX_DELTA_DISTANCE_PROFILE_EVALUATIONS
                ),
                "max_feasible_set_memberships": MAX_DELTA_MEMBERSHIPS,
                "max_ground_label_utf8_bytes": MAX_DELTA_LABEL_BYTES,
                "max_symmetric_exchange_candidate_checks_per_replay": (
                    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS
                ),
            },
        }
    )

    delta_matroid: FiniteDeltaMatroid = Field(
        description=(
            "Complete canonical finite delta-matroid. The source axiom replay "
            f"is bounded by {MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS} candidate "
            "checks; profile computation separately admits subset states and "
            "subset/feasible-row evaluations."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def preflight_ground_axis(cls, data: object) -> object:
        if isinstance(data, Mapping):
            source = data.get("delta_matroid")
            if isinstance(source, Mapping):
                ground = source.get("ground")
                if (
                    isinstance(ground, (list, tuple))
                    and len(ground) > MAX_DELTA_DISTANCE_PROFILE_STATES.bit_length() - 1
                ):
                    raise _validation_error(
                        "distance_profile_states_exceeded",
                        "the complete ground-subset profile exceeds the subset-state envelope",
                    )
        return data


def _preflight_extremal_input(data: object) -> object:
    """Bound nested source values before Pydantic materializes tuple fields."""
    if not isinstance(data, Mapping):
        return data
    raw = data.get("delta_matroid")
    if not isinstance(raw, Mapping):
        return data
    ground = raw.get("ground")
    if isinstance(ground, (list, tuple)):
        if len(ground) > MAX_DELTA_EXTREMAL_SOURCE_GROUND_LABELS:
            raise _validation_error(
                "source_ground_bound",
                "source ground axis exceeds the bounded conversion request size",
            )
        label_bytes = 0
        for label in ground:
            if isinstance(label, str):
                if len(label) > MAX_DELTA_LABEL_BYTES:
                    raise _validation_error(
                        "source_label_bound",
                        "source ground labels exceed the UTF-8 byte limit",
                    )
                try:
                    label_bytes += len(label.encode("utf-8"))
                except UnicodeEncodeError:
                    continue  # The canonical source model reports invalid text.
                if label_bytes > MAX_DELTA_LABEL_BYTES:
                    raise _validation_error(
                        "source_label_bound",
                        "source ground labels exceed the UTF-8 byte limit",
                    )
    rows = raw.get("feasible")
    if isinstance(rows, (list, tuple)):
        # Any structurally valid distinct family under the membership cap has
        # at most one empty row plus one row per membership.
        if len(rows) > MAX_DELTA_MEMBERSHIPS + 1:
            raise _validation_error("source_row_bound", "too many feasible rows")
        memberships = 0
        for row in rows:
            if isinstance(row, (list, tuple)):
                memberships += len(row)
                if memberships > MAX_DELTA_MEMBERSHIPS:
                    raise _validation_error(
                        "source_membership_bound", "too many feasible-set memberships"
                    )
    return data


class _DeltaMatroidExtremalRequest(StrictModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Convert a recognized delta-matroid to its complete lower or "
                "upper basis family. Source recognition uses the delta-matroid "
                "membership, UTF-8 label, and exchange-work limits; output "
                "uses the finite-basis ground, row, membership, label, and "
                "exchange-work limits."
            ),
            "admission_limits": {
                "max_source_feasible_set_memberships": MAX_DELTA_MEMBERSHIPS,
                "max_source_ground_labels_preparse": MAX_DELTA_EXTREMAL_SOURCE_GROUND_LABELS,
                "max_source_ground_label_utf8_bytes": MAX_DELTA_LABEL_BYTES,
                "max_source_exchange_candidate_checks": MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS,
                "max_output_ground_elements": MAX_FINITE_BASIS_GROUND_SIZE,
                "max_output_ground_label_utf8_bytes_each": MAX_FINITE_BASIS_LABEL_BYTES,
                "max_output_ground_label_utf8_bytes_total": MAX_FINITE_BASIS_TOTAL_LABEL_BYTES,
                "max_output_basis_rows": MAX_FINITE_BASIS_COUNT,
                "max_output_basis_memberships": MAX_FINITE_BASIS_MEMBERSHIPS,
                "max_output_basis_exchange_candidate_checks": MAX_FINITE_BASIS_EXCHANGE_CHECKS,
            },
        }
    )
    delta_matroid: FiniteDeltaMatroid

    @model_validator(mode="before")
    @classmethod
    def preflight_source(cls, data: object) -> object:
        return _preflight_extremal_input(data)


class DeltaMatroidLowerMatroidRequest(_DeltaMatroidExtremalRequest):
    """Compute the matroid of minimum-cardinality feasible sets."""


class DeltaMatroidUpperMatroidRequest(_DeltaMatroidExtremalRequest):
    """Compute the matroid of maximum-cardinality feasible sets."""


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"delta_matroid.{reason}", message)


class DeltaMatroidFromFeasibleSetsRequest(StrictModel):
    """One complete feasible family to recognize as a delta-matroid."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Recognize one complete feasible family by exhaustive "
                "symmetric exchange. Delta-matroid admission is fully "
                "result-sensitive: there are no separate ground-size or "
                "row-count caps, and the shared finite feasible-set carrier "
                "is structural only; the derived membership, UTF-8 label-byte, "
                "candidate-work bounds below "
                "admit every complete family whose recognition fits."
            ),
            "admission_limits": {
                "max_feasible_set_memberships": MAX_DELTA_MEMBERSHIPS,
                "max_ground_label_utf8_bytes": MAX_DELTA_LABEL_BYTES,
                "max_symmetric_exchange_candidate_checks_per_replay": (
                    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS
                ),
            },
        }
    )

    system: FiniteFeasibleSetSystem = Field(
        description=(
            "Complete labelled feasible-set family. Its delta-matroid admission "
            f"allows at most {MAX_DELTA_MEMBERSHIPS} total feasible-row "
            f"memberships, {MAX_DELTA_LABEL_BYTES} UTF-8 ground-label bytes, and "
            f"{MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS} symmetric-exchange candidate "
            "checks in its single complete axiom pass."
        )
    )


class DeltaMatroidRecognitionResult(StrictModel):
    """An exact delta-matroid value or the first complete axiom obstruction."""

    source: FiniteFeasibleSetSystem
    status: Literal["DELTA_MATROID", "NOT_A_DELTA_MATROID"]
    delta_matroid: FiniteDeltaMatroid | None = None
    obstruction: DeltaMatroidObstruction | None = None

    @model_validator(mode="after")
    def require_branch_consistency(self) -> Self:
        valid = (
            self.status == "DELTA_MATROID"
            and self.delta_matroid is not None
            and self.obstruction is None
        ) or (
            self.status == "NOT_A_DELTA_MATROID"
            and self.delta_matroid is None
            and self.obstruction is not None
        )
        if not valid:
            raise _validation_error(
                "status_branch",
                "status must agree with its retained value or obstruction",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        system: FiniteFeasibleSetSystem,
        *,
        delta_matroid: FiniteDeltaMatroid | None = None,
        obstruction: DeltaMatroidObstruction | None = None,
    ) -> Self:
        return cls.model_construct(
            source=system,
            status="DELTA_MATROID"
            if delta_matroid is not None
            else "NOT_A_DELTA_MATROID",
            delta_matroid=delta_matroid,
            obstruction=obstruction,
        )


__all__ = [
    "DeltaMatroidDistanceProfileRequest",
    "DeltaMatroidFromFeasibleSetsRequest",
    "DeltaMatroidLowerMatroidRequest",
    "DeltaMatroidRecognitionResult",
    "DeltaMatroidTwistRequest",
    "DeltaMatroidTwistResult",
    "DeltaMatroidUpperMatroidRequest",
    "DeltaMatroidWidthRequest",
    "DeltaMatroidWidthResult",
]
