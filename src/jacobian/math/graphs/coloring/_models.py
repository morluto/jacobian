"""Typed wire contracts for graph coloring and independent set operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel


class GraphEdgeList(StrictModel):
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


class KColorabilityRequest(StrictModel):
    graph: GraphEdgeList
    colors: int = Field(ge=1, le=20)


class KColorabilityResult(StrictModel):
    colorable: bool
    coloring: tuple[int, ...] | None = None
    vertex_count: int = Field(ge=1, le=20)
    colors: int = Field(ge=1, le=20)


class MaximalIndependentSetRequest(StrictModel):
    """One canonical candidate set in a bounded simple graph."""

    graph: GraphEdgeList
    candidate_set: tuple[int, ...] = Field(max_length=20)

    @model_validator(mode="after")
    def require_canonical_candidate_set(self) -> Self:
        if tuple(sorted(self.candidate_set)) != self.candidate_set:
            raise ValueError("candidate_set must be strictly increasing")
        if len(set(self.candidate_set)) != len(self.candidate_set):
            raise ValueError("candidate_set must not contain duplicate vertices")
        if any(
            vertex < 0 or vertex >= self.graph.vertex_count
            for vertex in self.candidate_set
        ):
            raise ValueError("candidate vertices must lie in 0..vertex_count-1")
        return self


class MaximalIndependentSetResult(StrictModel):
    """A closed decision with a concrete rejection witness when applicable."""

    decision: Literal["MAXIMAL", "NOT_INDEPENDENT", "INDEPENDENT_NOT_MAXIMAL"]
    blocking_edge: tuple[int, int] | None = None
    addable_vertex: int | None = None

    @model_validator(mode="after")
    def bind_witness_to_decision(self) -> Self:
        if self.decision == "MAXIMAL":
            if self.blocking_edge is not None or self.addable_vertex is not None:
                raise ValueError("a maximal result must not carry a rejection witness")
            return self
        if self.decision == "NOT_INDEPENDENT":
            if self.blocking_edge is None or self.addable_vertex is not None:
                raise ValueError(
                    "a non-independent result requires exactly one blocking edge"
                )
            u, v = self.blocking_edge
            if u < 0 or v < 0 or u >= v:
                raise ValueError("blocking_edge must be a canonical pair u < v")
            return self
        if self.blocking_edge is not None or self.addable_vertex is None:
            raise ValueError(
                "an independent non-maximal result requires exactly one addable vertex"
            )
        if self.addable_vertex < 0:
            raise ValueError("addable_vertex must be nonnegative")
        return self
