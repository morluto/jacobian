"""Typed wire contracts for finite delta-matroid recognition."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.delta_matroids.values import (
    MAX_DELTA_AXIOM_REPLAYS_PER_REQUEST,
    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS,
    MAX_DELTA_LABEL_BYTES,
    MAX_DELTA_MEMBERSHIPS,
    MAX_DELTA_RESULT_BYTES,
    DeltaMatroidObstruction,
    FiniteDeltaMatroid,
    canonical_feasible_rows,
    first_symmetric_exchange_obstruction,
    require_delta_matroid_admission,
)
from jacobian.math.greedoids.values import FiniteFeasibleSetSystem


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
                "candidate-work, replay-count, and result-size bounds below "
                "admit every complete family whose recognition fits."
            ),
            "admission_limits": {
                "max_feasible_set_memberships": MAX_DELTA_MEMBERSHIPS,
                "max_ground_label_utf8_bytes": MAX_DELTA_LABEL_BYTES,
                "max_symmetric_exchange_candidate_checks_per_replay": (
                    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS
                ),
                "max_complete_axiom_replays_per_request": (
                    MAX_DELTA_AXIOM_REPLAYS_PER_REQUEST
                ),
                "max_result_bytes": MAX_DELTA_RESULT_BYTES,
            },
        }
    )

    system: FiniteFeasibleSetSystem = Field(
        description=(
            "Complete labelled feasible-set family. Its delta-matroid admission "
            f"allows at most {MAX_DELTA_MEMBERSHIPS} total feasible-row "
            f"memberships, {MAX_DELTA_LABEL_BYTES} UTF-8 ground-label bytes, and "
            f"{MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS} symmetric-exchange candidate "
            f"checks per complete axiom replay; one recognized request performs "
            f"at most {MAX_DELTA_AXIOM_REPLAYS_PER_REQUEST} complete replays, "
            f"and the serialized recognition result is at most "
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
            # The declared delta_matroid already passed its own complete
            # defining-invariant replay during field validation; binding only
            # has to pin it to the retained source's canonical wire order.
            if (
                self.delta_matroid is None
                or self.delta_matroid.ground != self.source.ground
                or self.delta_matroid.feasible != canonical_feasible_rows(self.source)
            ):
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
