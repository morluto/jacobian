"""Typed wire contracts for finite delta-matroid recognition."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.delta_matroids.values import (
    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS,
    MAX_DELTA_FEASIBLE_SETS,
    MAX_DELTA_LABEL_BYTES,
    MAX_DELTA_MEMBERSHIPS,
    MAX_DELTA_RESULT_BYTES,
    DeltaMatroidObstruction,
    FiniteDeltaMatroid,
    canonical_delta_matroid,
    first_symmetric_exchange_obstruction,
    require_delta_matroid_admission,
)
from jacobian.math.greedoids.values import MAX_GROUND_SIZE, FiniteFeasibleSetSystem


class DeltaMatroidFromFeasibleSetsRequest(StrictModel):
    """One complete feasible family to recognize as a delta-matroid."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Recognize one complete feasible family by exhaustive symmetric "
                "exchange. Delta-matroid admission is result-sensitive: there is "
                "no separate delta-specific ground-size cap; the shared finite "
                "feasible-set carrier bounds ground labels to 64 elements."
            ),
            "admission_limits": {
                "max_ground_elements": MAX_GROUND_SIZE,
                "max_feasible_sets": MAX_DELTA_FEASIBLE_SETS,
                "max_feasible_set_memberships": MAX_DELTA_MEMBERSHIPS,
                "max_ground_label_utf8_bytes": MAX_DELTA_LABEL_BYTES,
                "max_symmetric_exchange_candidate_checks": (
                    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS
                ),
                "max_result_bytes": MAX_DELTA_RESULT_BYTES,
            },
        }
    )

    system: FiniteFeasibleSetSystem = Field(
        description=(
            "Complete labelled feasible-set family. Its delta-matroid admission "
            f"allows at most {MAX_DELTA_FEASIBLE_SETS} feasible rows, "
            f"{MAX_DELTA_MEMBERSHIPS} total memberships, "
            f"{MAX_DELTA_LABEL_BYTES} UTF-8 ground-label bytes, and "
            f"{MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS} symmetric-exchange candidate "
            f"checks; the serialized recognition result is at most "
            f"{MAX_DELTA_RESULT_BYTES} bytes."
        )
    )

    @model_validator(mode="after")
    def require_bounded_exchange_replay(self) -> Self:
        require_delta_matroid_admission(self.system)
        return self


class DeltaMatroidRecognitionResult(StrictModel):
    """An exact delta-matroid value or the first complete axiom obstruction."""

    source: FiniteFeasibleSetSystem
    status: Literal["DELTA_MATROID", "NOT_A_DELTA_MATROID"]
    delta_matroid: FiniteDeltaMatroid | None = None
    obstruction: DeltaMatroidObstruction | None = None

    @model_validator(mode="after")
    def bind_result_to_complete_source_replay(self) -> Self:
        require_delta_matroid_admission(self.source)
        expected_obstruction = first_symmetric_exchange_obstruction(self.source)
        if expected_obstruction is None:
            if self.status != "DELTA_MATROID":
                raise ValueError("a valid feasible family must return DELTA_MATROID")
            if self.obstruction is not None:
                raise ValueError("a valid delta-matroid result has no obstruction")
            expected_value = canonical_delta_matroid(self.source)
            if self.delta_matroid != expected_value:
                raise ValueError(
                    "delta_matroid must equal the canonical replay of the retained source"
                )
            return self
        if self.status != "NOT_A_DELTA_MATROID":
            raise ValueError("an exchange obstruction must return NOT_A_DELTA_MATROID")
        if self.delta_matroid is not None:
            raise ValueError(
                "an invalid feasible family must not carry a delta-matroid"
            )
        if self.obstruction != expected_obstruction:
            raise ValueError("obstruction must equal the first source exchange failure")
        return self


__all__ = [
    "DeltaMatroidFromFeasibleSetsRequest",
    "DeltaMatroidRecognitionResult",
]
