"""Typed contracts for word-cube combinatorial-line hypergraph operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

MAX_ALPHABET_SIZE = 10
MAX_DIMENSION = 6
MAX_VERTICES = 10_000


class CombinatorialLineHypergraphRequest(StrictModel):
    """Request to construct the combinatorial-line hypergraph of [q]^d."""

    alphabet_size: int = Field(ge=2, le=MAX_ALPHABET_SIZE)
    dimension: int = Field(ge=1, le=MAX_DIMENSION)

    @model_validator(mode="after")
    def validate_vertex_count(self) -> Self:
        vertex_count = self.alphabet_size**self.dimension
        if vertex_count > MAX_VERTICES:
            raise PydanticCustomError(
                "word_cube.vertex_count_exceeds_bound",
                f"q^d = {vertex_count} exceeds the {MAX_VERTICES}-vertex limit",
            )
        return self


class CombinatorialLineHypergraphResult(StrictModel):
    """The combinatorial-line hypergraph of [q]^d."""

    alphabet_size: int
    dimension: int
    hypergraph: FiniteHypergraph


__all__ = [
    "MAX_ALPHABET_SIZE",
    "MAX_DIMENSION",
    "MAX_VERTICES",
    "CombinatorialLineHypergraphRequest",
    "CombinatorialLineHypergraphResult",
]
