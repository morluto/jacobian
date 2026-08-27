"""Typed contracts for the Sidon extension-profile operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictBool, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics._difference_set_models import (
    AdditiveInteger,
    MAX_ADDITIVE_INTEGER_LENGTH,
    MAX_SIDON_SET_SIZE,
    _difference_set_validation_error,
)

MAX_EXTENSION_SOURCE_SIZE = 32
MAX_EXTENSION_CANDIDATES = 1000


class SidonExtensionProfileRequest(StrictModel):
    """Source Sidon set A and candidate set C disjoint from A."""

    source_elements: tuple[AdditiveInteger, ...] = Field(
        max_length=MAX_EXTENSION_SOURCE_SIZE
    )
    candidate_elements: tuple[AdditiveInteger, ...] = Field(
        max_length=MAX_EXTENSION_CANDIDATES
    )

    @model_validator(mode="after")
    def require_unique_and_disjoint(self) -> Self:
        if len(set(self.source_elements)) != len(self.source_elements):
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant",
                "source elements must be unique",
            )
        if len(set(self.candidate_elements)) != len(self.candidate_elements):
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant",
                "candidate elements must be unique",
            )
        source_set = set(self.source_elements)
        candidate_set = set(self.candidate_elements)
        if source_set & candidate_set:
            raise _difference_set_validation_error(
                "combinatorics.sidon_invariant",
                "source and candidate sets must be disjoint",
            )
        return self


class SidonExtensionObstruction(StrictModel):
    """One replayable repeated-difference obstruction for a rejected candidate."""

    candidate: AdditiveInteger
    repeated_difference: str
    pair_a: tuple[AdditiveInteger, AdditiveInteger]
    pair_b: tuple[AdditiveInteger, AdditiveInteger]


class SidonExtensionCandidateResult(StrictModel):
    """One candidate's admissibility or obstruction."""

    candidate: AdditiveInteger
    is_admissible: StrictBool
    obstruction: SidonExtensionObstruction | None = None


class SidonExtensionProfileResult(StrictModel):
    """Complete partition of candidates into admissible and rejected."""

    source_elements: tuple[AdditiveInteger, ...]
    candidate_elements: tuple[AdditiveInteger, ...]
    admissible: list[AdditiveInteger]
    rejected: list[SidonExtensionCandidateResult]


__all__ = [
    "MAX_EXTENSION_SOURCE_SIZE",
    "MAX_EXTENSION_CANDIDATES",
    "SidonExtensionProfileRequest",
    "SidonExtensionProfileResult",
    "SidonExtensionCandidateResult",
    "SidonExtensionObstruction",
]
