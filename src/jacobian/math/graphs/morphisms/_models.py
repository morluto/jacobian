"""Typed wire contracts for graph morphism operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_VERTICES = 64
MAX_EDGES = 512

# Exhaustive backtracking search over graph morphisms is exponential in the
# vertex count.  This dedicated bound keeps every search-based morphism
# operation inside a tested, provably bounded domain.
MORPHISM_MAX_VERTICES = 20


class SimpleGraph(StrictModel):
    """A simple undirected graph with integer-labelled vertices."""

    vertex_count: int = Field(ge=1, le=MAX_VERTICES)
    edges: tuple[tuple[int, int], ...] = Field(default=(), max_length=MAX_EDGES)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for u, v in self.edges:
            if not (0 <= u < self.vertex_count and 0 <= v < self.vertex_count):
                raise ValueError("edge vertices must be in 0..vertex_count-1")
            if u == v:
                raise ValueError("self-loops are not allowed")
            endpoint_pair = (min(u, v), max(u, v))
            if endpoint_pair in seen:
                raise ValueError("edges must be unique")
            seen.add(endpoint_pair)
        return self


class HomomorphismCheckRequest(StrictModel):
    source_graph: SimpleGraph
    target_graph: SimpleGraph
    vertex_map: tuple[int, ...]

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if len(self.vertex_map) != self.source_graph.vertex_count:
            raise ValueError("vertex_map length must match source_graph vertex_count")
        if any(not 0 <= v < self.target_graph.vertex_count for v in self.vertex_map):
            raise ValueError("vertex_map entries must be valid target_graph vertices")
        return self


class HomomorphismFindRequest(StrictModel):
    source_graph: SimpleGraph
    target_graph: SimpleGraph

    @model_validator(mode="after")
    def require_search_bounded(self) -> Self:
        if self.source_graph.vertex_count > MORPHISM_MAX_VERTICES:
            raise ValueError(
                f"source graph must have at most {MORPHISM_MAX_VERTICES} vertices"
            )
        return self


class CoreCheckRequest(StrictModel):
    graph: SimpleGraph

    @model_validator(mode="after")
    def require_search_bounded(self) -> Self:
        if self.graph.vertex_count > MORPHISM_MAX_VERTICES:
            raise ValueError(
                f"graph must have at most {MORPHISM_MAX_VERTICES} vertices"
            )
        return self


class RetractionCheckRequest(StrictModel):
    graph: SimpleGraph
    subgraph_vertices: tuple[int, ...]

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if self.graph.vertex_count > MORPHISM_MAX_VERTICES:
            raise ValueError(
                f"graph must have at most {MORPHISM_MAX_VERTICES} vertices"
            )
        if len(self.subgraph_vertices) > self.graph.vertex_count:
            raise ValueError("subgraph_vertices must be a subset")
        for v in self.subgraph_vertices:
            if not 0 <= v < self.graph.vertex_count:
                raise ValueError("subgraph_vertices must be valid vertex indices")
        if len(set(self.subgraph_vertices)) != len(self.subgraph_vertices):
            raise ValueError("subgraph_vertices must be unique")
        return self


class HomomorphismCheckResult(StrictModel):
    is_homomorphism: bool
    method: str = "EDGE_PRESERVING_CHECK"


class HomomorphismFindResult(StrictModel):
    found: bool
    vertex_map: tuple[int, ...] = ()
    method: str = "BACKTRACKING_SEARCH"


class CoreCheckResult(StrictModel):
    is_core: bool
    method: str = "ENDOMORPHISM_CHECK"


class RetractionCheckResult(StrictModel):
    is_retraction: bool
    method: str = "HOMOMORPHISM_CHECK"
