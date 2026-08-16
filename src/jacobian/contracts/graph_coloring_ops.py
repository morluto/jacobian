"""Typed wire contracts for graph coloring and independent set operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel


class GraphEdgeList(ContractModel):
    """A simple undirected graph given by an edge list."""

    # Exact SAT instances are deliberately kept small enough for one direct
    # solver call in the stateless server.
    vertex_count: int = Field(ge=1, le=20)
    edges: tuple[tuple[int, int], ...] = Field(
        max_length=512,
    )

    @model_validator(mode="after")
    def require_valid_edges(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for u, v in self.edges:
            if not (0 <= u < self.vertex_count and 0 <= v < self.vertex_count):
                raise ValueError("edge vertices must be in 0..vertex_count-1")
            if u == v:
                raise ValueError("a simple graph cannot contain self-loops")
            edge = (min(u, v), max(u, v))
            if edge in seen:
                raise ValueError("a simple graph cannot contain duplicate edges")
            seen.add(edge)
        return self


class KColorabilityRequest(ContractModel):
    graph: GraphEdgeList
    colors: int = Field(ge=1, le=20)


class KColorabilityResult(ContractModel):
    colorable: bool
    coloring: tuple[int, ...] | None = None
    vertex_count: int = Field(ge=1, le=20)
    colors: int = Field(ge=1, le=20)


class MaximumIndependentSetRequest(ContractModel):
    graph: GraphEdgeList


class MaximumIndependentSetResult(ContractModel):
    independent_set: tuple[int, ...]
    cardinality: int = Field(ge=0, le=20)


class MaximalIndependentSetRequest(ContractModel):
    """Decide whether a given vertex set is a maximal independent set."""

    graph: GraphEdgeList
    candidate_set: tuple[int, ...] = Field(max_length=20)

    @model_validator(mode="after")
    def require_valid_candidate(self) -> Self:
        seen: set[int] = set()
        for vertex in self.candidate_set:
            if not (0 <= vertex < self.graph.vertex_count):
                raise ValueError("candidate vertices must lie in 0..vertex_count-1")
            if vertex in seen:
                raise ValueError("candidate set must contain no duplicate vertices")
            seen.add(vertex)
        return self


class MaximalIndependentSetResult(ContractModel):
    """Decision outcome for a candidate vertex set."""

    decision: Literal["MAXIMAL", "NOT_INDEPENDENT", "INDEPENDENT_NOT_MAXIMAL"]
    candidate_set: tuple[int, ...]
    vertex_count: int = Field(ge=1, le=20)
