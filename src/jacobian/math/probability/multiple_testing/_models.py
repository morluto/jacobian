"""Typed wire contracts for multiple testing operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math._labels import OpaqueLabel

MAX_HYPOTHESES = 1000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by multiple-testing contracts."""

    return PydanticCustomError(f"multiple_testing.{reason}", message)


class HypothesisSpec(StrictModel):
    """One labelled p-value."""

    hypothesis_id: OpaqueLabel
    p_value: CanonicalRational

    @model_validator(mode="after")
    def require_probability(self) -> Self:
        value = self.p_value.as_fraction()
        if not 0 <= value <= 1:
            raise _validation_error("p_value_out_of_range", "p-value must be in [0, 1]")
        return self


class BHStepUpRequest(StrictModel):
    """Benjamini-Hochberg step-up procedure."""

    hypotheses: tuple[HypothesisSpec, ...] = Field(
        min_length=1, max_length=MAX_HYPOTHESES
    )
    level: CanonicalRational

    @model_validator(mode="after")
    def require_unique_ids_and_probability_level(self) -> Self:
        ids = [h.hypothesis_id for h in self.hypotheses]
        if len(ids) != len(set(ids)):
            raise _validation_error(
                "hypothesis_ids_not_unique", "hypothesis IDs must be unique"
            )
        level = self.level.as_fraction()
        if not 0 <= level <= 1:
            raise _validation_error("level_out_of_range", "level must be in [0, 1]")
        return self


class BHStepUpResult(StrictModel):
    """BH step-up rejection set."""

    critical_index: int = Field(ge=0)
    cutoff_threshold: str
    rejected: tuple[OpaqueLabel, ...]
    total_hypotheses: int = Field(ge=1)


class FDPRequest(StrictModel):
    """False discovery proportion computation."""

    rejected_ids: tuple[OpaqueLabel, ...] = Field(default=())
    true_null_ids: tuple[OpaqueLabel, ...] = Field(default=())


class FDPResult(StrictModel):
    """Exact false discovery proportion."""

    false_discoveries: int = Field(ge=0)
    total_rejections: int = Field(ge=0)
    fdp: str


__all__ = [
    "BHStepUpRequest",
    "BHStepUpResult",
    "FDPRequest",
    "FDPResult",
    "HypothesisSpec",
]
