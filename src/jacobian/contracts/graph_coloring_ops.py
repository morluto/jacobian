"""Typed wire contracts for graph coloring and independent set operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalInteger


class GraphEdgeList(ContractModel):
    """A simple undirected graph given by an edge list."""

    vertex_count: int = Field(ge=1, le=64)
    edges: tuple[tuple[int, int], ...] = Field(
        max_length=512,
    )

    @model_validator(mode="after")
    def require_valid_edges(self) -> Self:
        for u, v in self.edges:
            if not (0 <= u < self.vertex_count and 0 <= v < self.vertex_count):
                raise ValueError("edge vertices must be in 0..vertex_count-1")
        return self


class KColorabilityRequest(ContractModel):
    graph: GraphEdgeList
    colors: int = Field(ge=1, le=64)


class KColorabilityResult(ContractModel):
    colorable: bool
    coloring: tuple[int, ...] | None = None
    vertex_count: int = Field(ge=1, le=64)
    colors: int = Field(ge=1, le=64)


class MaximumIndependentSetRequest(ContractModel):
    graph: GraphEdgeList


class MaximumIndependentSetResult(ContractModel):
    independent_set: tuple[int, ...]
    cardinality: int = Field(ge=0, le=64)
