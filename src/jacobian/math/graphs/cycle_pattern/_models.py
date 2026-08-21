"""Typed wire contracts for graph cycle and subgraph-pattern operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel


class UndirectedGraph(StrictModel):
    """A simple undirected graph for cycle and pattern operations."""

    vertex_count: int = Field(ge=1, le=64)
    edges: tuple[tuple[int, int], ...] = Field(min_length=0, max_length=512)

    @model_validator(mode="after")
    def require_valid_edges(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for source, target in self.edges:
            if not (
                0 <= source < self.vertex_count
                and 0 <= target < self.vertex_count
            ):
                raise ValueError("edge vertices must be in 0..vertex_count-1")
            if source == target:
                raise ValueError("self-loops are not allowed")
            canonical = (min(source, target), max(source, target))
            if canonical in seen:
                raise ValueError("undirected edges must be unique")
            seen.add(canonical)
        return self


class FixedLengthCycleRequest(StrictModel):
    """Decide whether a graph contains a simple cycle of a given length."""

    graph: UndirectedGraph
    length: int = Field(ge=3, le=20)

    @model_validator(mode="after")
    def require_length_within_bounds(self) -> Self:
        if self.length > self.graph.vertex_count:
            raise ValueError("cycle length cannot exceed vertex count")
        return self


class FixedLengthCycleResult(StrictModel):
    """Whether a simple k-cycle exists, with an explicit witness."""

    vertex_count: int = Field(ge=1, le=64)
    length: int = Field(ge=3, le=20)
    exists: bool
    cycle: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def bind_witness(self) -> Self:
        if self.exists is (self.cycle is None):
            raise ValueError("exactly an existing cycle carries one witness")
        if self.cycle is not None:
            if len(self.cycle) != self.length:
                raise ValueError("witness cycle length must match the declared length")
            if len(set(self.cycle)) != self.length:
                raise ValueError("witness cycle vertices must be distinct")
        return self


class SubgraphPatternRequest(StrictModel):
    """Find an injective embedding of a pattern graph into a host graph."""

    host: UndirectedGraph
    pattern: UndirectedGraph

    @model_validator(mode="after")
    def require_pattern_fits(self) -> Self:
        if self.pattern.vertex_count > self.host.vertex_count:
            raise ValueError("pattern vertex count cannot exceed host vertex count")
        return self


class SubgraphEmbedding(StrictModel):
    """One injective embedding of a pattern graph into a host graph."""

    mapping: tuple[tuple[int, int], ...]

    @model_validator(mode="after")
    def require_valid_mapping(self) -> Self:
        if not self.mapping:
            raise ValueError("embedding mapping must be nonempty")
        domain = tuple(src for src, _ in self.mapping)
        codomain = tuple(dst for _, dst in self.mapping)
        if domain != tuple(sorted(domain)):
            raise ValueError("embedding domain must be sorted")
        if len(set(codomain)) != len(codomain):
            raise ValueError("embedding codomain must be injective")
        return self


class SubgraphPatternResult(StrictModel):
    """Whether a subgraph embedding exists, with an explicit witness."""

    host_vertex_count: int = Field(ge=1, le=64)
    pattern_vertex_count: int = Field(ge=1, le=64)
    exists: bool
    embedding: SubgraphEmbedding | None = None

    @model_validator(mode="after")
    def bind_witness(self) -> Self:
        if self.exists is (self.embedding is None):
            raise ValueError("exactly an existing embedding carries one witness")
        if self.embedding is not None:
            if len(self.embedding.mapping) != self.pattern_vertex_count:
                raise ValueError("embedding must map every pattern vertex")
        return self


__all__ = [
    "FixedLengthCycleRequest",
    "FixedLengthCycleResult",
    "SubgraphEmbedding",
    "SubgraphPatternRequest",
    "SubgraphPatternResult",
    "UndirectedGraph",
]
