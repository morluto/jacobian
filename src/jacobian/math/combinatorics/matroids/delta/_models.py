"""Typed wire contracts for finite delta-matroid recognition."""

from __future__ import annotations

import json
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids.delta.values import (
    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS,
    MAX_DELTA_LABEL_BYTES,
    MAX_DELTA_MEMBERSHIPS,
    MAX_DELTA_TWIST_RESULT_BYTES,
    DeltaMatroidObstruction,
    FiniteDeltaMatroid,
)


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
                f" The source-bound result is limited to {MAX_DELTA_TWIST_RESULT_BYTES}"
                " compact JSON bytes."
            ),
            "admission_limits": {
                "max_feasible_set_memberships": MAX_DELTA_MEMBERSHIPS,
                "max_ground_label_utf8_bytes": MAX_DELTA_LABEL_BYTES,
                "max_symmetric_exchange_candidate_checks_per_replay": (
                    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS
                ),
                "max_source_bound_result_compact_json_bytes": (
                    MAX_DELTA_TWIST_RESULT_BYTES
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
            "checks before using the source family. The source-bound result is "
            f"limited to {MAX_DELTA_TWIST_RESULT_BYTES} compact JSON bytes."
        )
    )
    subset: tuple[int, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical_subset(self) -> Self:
        require_twist_subset(self.delta_matroid, self.subset)
        return self


class DeltaMatroidTwistResult(DeltaMatroidTwistRequest):
    """The source, twist subset, and exact target feasible family."""

    twisted: FiniteDeltaMatroid

    @model_validator(mode="after")
    def require_same_ground_axis(self) -> Self:
        if self.twisted.ground != self.delta_matroid.ground:
            raise _validation_error(
                "twist_ground_axis",
                "twisted delta-matroid must preserve the source ground axis",
            )
        if len(self.twisted.feasible) != len(self.delta_matroid.feasible):
            raise _validation_error(
                "twist_family_cardinality",
                "twisting must preserve the number of feasible sets",
            )
        return self

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


def twist_result_serialized_bytes(
    delta_matroid: FiniteDeltaMatroid,
    subset: tuple[int, ...],
    twisted: FiniteDeltaMatroid,
) -> int:
    """Count the compact JSON bytes of a source-bound twist result exactly."""

    source_json = delta_matroid.model_dump_json()
    subset_json = json.dumps(subset, separators=(",", ":"))
    twisted_json = twisted.model_dump_json()
    result_json = (
        '{"delta_matroid":'
        + source_json
        + ',"subset":'
        + subset_json
        + ',"twisted":'
        + twisted_json
        + "}"
    )
    return len(result_json.encode("utf-8"))


class DeltaMatroidWidthRequest(StrictModel):
    """Compute the width (largest minus smallest feasible-set size)."""

    delta_matroid: FiniteDeltaMatroid


class DeltaMatroidWidthResult(DeltaMatroidWidthRequest):
    width: int = Field(ge=0)

    @classmethod
    def _from_kernel(cls, delta_matroid: FiniteDeltaMatroid, width: int) -> Self:
        return cls.model_construct(delta_matroid=delta_matroid, width=width)


class DeltaMatroidDistanceRequest(StrictModel):
    """Distance from one ground subset to the complete feasible family."""

    delta_matroid: FiniteDeltaMatroid
    subset: tuple[int, ...] = Field(description="Sorted distinct ground indices")

    @model_validator(mode="after")
    def canonical_subset(self) -> Self:
        require_twist_subset(self.delta_matroid, self.subset)
        return self


class DeltaMatroidDistanceResult(DeltaMatroidDistanceRequest):
    """Exact Hamming distance and a deterministic nearest feasible set."""

    distance: int = Field(ge=0)
    nearest_feasible: tuple[int, ...]

    @classmethod
    def _from_kernel(
        cls,
        delta_matroid: FiniteDeltaMatroid,
        subset: tuple[int, ...],
        distance: int,
        nearest_feasible: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            delta_matroid=delta_matroid,
            subset=subset,
            distance=distance,
            nearest_feasible=nearest_feasible,
        )


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
    "DeltaMatroidDistanceRequest",
    "DeltaMatroidDistanceResult",
    "DeltaMatroidFromFeasibleSetsRequest",
    "DeltaMatroidRecognitionResult",
    "DeltaMatroidTwistRequest",
    "DeltaMatroidTwistResult",
    "DeltaMatroidWidthRequest",
    "DeltaMatroidWidthResult",
    "twist_result_serialized_bytes",
]
