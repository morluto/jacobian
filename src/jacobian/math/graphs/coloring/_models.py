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


# ---------------------------------------------------------------------------
# Edge coloring
# ---------------------------------------------------------------------------


class EdgeKColorabilityRequest(StrictModel):
    """Decide whether a simple graph admits a proper ``k``-edge-coloring."""

    graph: GraphEdgeList
    colors: int = Field(ge=1, le=20)


class EdgeKColorabilityResult(StrictModel):
    """Whether a proper ``k``-edge-coloring exists, with one coloring witness."""

    colorable: bool
    coloring: tuple[int, ...] | None = None
    edge_count: int = Field(ge=0)
    colors: int = Field(ge=1, le=20)

    @model_validator(mode="after")
    def require_witness_consistency(self) -> Self:
        if self.colorable:
            if self.coloring is None:
                raise ValueError("a colorable result must carry a coloring witness")
            if len(self.coloring) != self.edge_count:
                raise ValueError("a coloring must assign one color per edge")
        else:
            if self.coloring is not None:
                raise ValueError("a non-colorable result must not carry a coloring")
        return self


class EdgeColoringCheckRequest(StrictModel):
    """Validate a submitted edge-to-color assignment as a proper edge coloring."""

    graph: GraphEdgeList
    colors: int = Field(ge=1, le=20)
    coloring: tuple[int, ...] = Field(min_length=0)

    @model_validator(mode="after")
    def require_assignment_length(self) -> Self:
        if len(self.coloring) != len(self.graph.edges):
            raise ValueError("coloring must assign one color per edge")
        for value in self.coloring:
            if type(value) is not int or not 0 <= value < self.colors:
                raise ValueError("coloring values must be in 0..colors-1")
        return self


class EdgeColoringCheckResult(StrictModel):
    """Whether a submitted edge coloring is proper, with a blocking edge."""

    proper: bool
    blocking_edge: tuple[int, int] | None = None

    @model_validator(mode="after")
    def require_blocking_edge_consistency(self) -> Self:
        if self.proper and self.blocking_edge is not None:
            raise ValueError("a proper coloring must not carry a blocking edge")
        if not self.proper and self.blocking_edge is None:
            raise ValueError("an improper coloring must carry a blocking edge")
        return self
