"""Typed wire contracts for finite delta-matroid recognition."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.delta_matroids.values import (
    DeltaMatroidObstruction,
    FiniteDeltaMatroid,
    canonical_delta_matroid,
    first_symmetric_exchange_obstruction,
    require_delta_matroid_admission,
)
from jacobian.math.greedoids.values import FiniteFeasibleSetSystem


class DeltaMatroidFromFeasibleSetsRequest(StrictModel):
    """One complete feasible family to recognize as a delta-matroid."""

    system: FiniteFeasibleSetSystem

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
