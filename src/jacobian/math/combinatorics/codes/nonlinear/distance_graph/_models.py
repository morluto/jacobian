"""Typed contracts for the binary code distance graph operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.codes.nonlinear.values import ExplicitBinaryCode
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph

MAX_CODE_SIZE = 256


class BinaryCodeDistanceGraphRequest(StrictModel):
    """Request for the Hamming distance graph of a binary code."""

    source: ExplicitBinaryCode
    target_distance: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.target_distance > self.source.length:
            raise PydanticCustomError(
                "code.distance_exceeds_length",
                "target_distance must not exceed code length",
            )
        if len(self.source.codewords) > MAX_CODE_SIZE:
            raise PydanticCustomError(
                "code.too_many_codewords",
                f"at most {MAX_CODE_SIZE} codewords are supported",
            )
        return self


class BinaryCodeDistanceGraphResult(StrictModel):
    """The Hamming distance graph of a binary code."""

    source: ExplicitBinaryCode
    target_distance: int
    graph: IndexedSimpleUndirectedGraph
    edge_count: int


__all__ = [
    "MAX_CODE_SIZE",
    "BinaryCodeDistanceGraphRequest",
    "BinaryCodeDistanceGraphResult",
]
