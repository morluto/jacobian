"""Typed wire contracts for finite delta-matroid recognition."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.greedoids.values import FiniteFeasibleSetSystem
from jacobian.math.combinatorics.matroids.delta.values import (
    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS,
    MAX_DELTA_LABEL_BYTES,
    MAX_DELTA_MEMBERSHIPS,
    DeltaMatroidObstruction,
    FiniteDeltaMatroid,
)


class DeltaMatroidTwistRequest(StrictModel):
    """Twist a delta-matroid by a canonical ground-index subset."""

    delta_matroid: FiniteDeltaMatroid
    subset: tuple[int, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical_subset(self) -> Self:
        if self.subset != tuple(sorted(set(self.subset))):
            raise _validation_error(
                "subset_not_canonical", "twist subset must be sorted and distinct"
            )
        if any(
            index < 0 or index >= len(self.delta_matroid.ground)
            for index in self.subset
        ):
            raise _validation_error(
                "subset_out_of_range", "twist subset index is outside the ground set"
            )
        return self


class DeltaMatroidTwistResult(DeltaMatroidTwistRequest):
    """The exact feasible family obtained by symmetric-difference twist."""

    twisted: FiniteDeltaMatroid

    @classmethod
    def _from_kernel(
        cls, request: DeltaMatroidTwistRequest, twisted: FiniteDeltaMatroid
    ) -> Self:
        return cls.model_construct(
            delta_matroid=request.delta_matroid,
            subset=request.subset,
            twisted=twisted,
        )


class DeltaMatroidWidthRequest(StrictModel):
    """Compute the width (largest minus smallest feasible-set size)."""

    delta_matroid: FiniteDeltaMatroid


class DeltaMatroidWidthResult(DeltaMatroidWidthRequest):
    width: int = Field(ge=0)

    @classmethod
    def _from_kernel(cls, request: DeltaMatroidWidthRequest, width: int) -> Self:
        return cls.model_construct(delta_matroid=request.delta_matroid, width=width)


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
    "DeltaMatroidFromFeasibleSetsRequest",
    "DeltaMatroidRecognitionResult",
    "DeltaMatroidTwistRequest",
    "DeltaMatroidTwistResult",
    "DeltaMatroidWidthRequest",
    "DeltaMatroidWidthResult",
]
