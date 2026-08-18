"""Typed wire contracts for frame operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_VECTORS = 32
MAX_DIM = 16
MAX_VALUE = 1000


class FrameRequest(StrictModel):
    """A finite frame: a set of vectors in R^d."""

    vectors: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=MAX_VECTORS)

    @model_validator(mode="after")
    def require_valid_frame(self) -> Self:
        dim = len(self.vectors[0])
        if dim == 0 or dim > MAX_DIM:
            raise ValueError("vector dimension must be between 1 and 16")
        if any(len(v) != dim for v in self.vectors):
            raise ValueError("all vectors must have equal dimension")
        if any(abs(x) > MAX_VALUE for v in self.vectors for x in v):
            raise ValueError("vector entries must be bounded")
        return self


class GramResult(StrictModel):
    gram: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=1)
    method: str = "DOT_PRODUCT"


class CoherenceResult(StrictModel):
    coherence: float
    method: str = "MAX_OFFDIAGONAL"


class FramePotentialResult(StrictModel):
    potential: float
    method: str = "SUM_OF_SQUARES"
