"""Typed wire contracts for finite delta-matroid recognition."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.delta_matroids.values import (
    MAX_DELTA_AXIOM_REPLAYS_PER_REQUEST,
    MAX_DELTA_EXCHANGE_CANDIDATE_CHECKS,
    MAX_DELTA_LABEL_BYTES,
    MAX_DELTA_MEMBERSHIPS,
    MAX_DELTA_RESULT_BYTES,
    DeltaMatroidAdmissionError,
    DeltaMatroidObstruction,
    FiniteDeltaMatroid,
    require_delta_matroid_admission,
)
from jacobian.math.greedoids.values import FiniteFeasibleSetSystem


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
        try:
            require_delta_matroid_admission(self.system)
        except DeltaMatroidAdmissionError as exc:
            raise _validation_error(exc.reason, str(exc)) from None
        return self


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
        return cls(
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
]
