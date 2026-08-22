"""Typed wire contracts for graph cycle and subgraph-pattern operations."""

from __future__ import annotations

from typing import Literal, Self

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


def _require_decided_witness_pair(
    exists: bool | None, witness: object
) -> None:
    """A decided search states a claim; exactly an affirmative one carries a witness."""
    if exists is None:
        raise ValueError("a decided search states whether the target exists")
    if exists is (witness is None):
        raise ValueError("exactly an existing witness carries one witness")


class SearchOutcome(StrictModel):
    """Shared outcome discriminator for bounded exhaustive searches."""

    outcome: Literal["DECIDED", "SEARCH_BUDGET_EXCEEDED"] = "DECIDED"
    detail: str | None = None

    @model_validator(mode="after")
    def require_outcome_shape(self) -> Self:
        if self.outcome == "DECIDED":
            if self.detail is not None:
                raise ValueError("decided searches carry no failure detail")
        elif self.detail is None:
            raise ValueError("an exceeded search budget carries a detail")
        return self


class FixedLengthCycleResult(SearchOutcome):
    """Whether a simple k-cycle exists, with an explicit witness.

    Carries its source graph so the witness replays against the exact edge
    relation instead of trusting vertex bookkeeping.
    """

    graph: UndirectedGraph
    vertex_count: int = Field(ge=1, le=64)
    length: int = Field(ge=3, le=20)
    exists: bool | None = None
    cycle: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def bind_witness(self) -> Self:
        if self.outcome != "DECIDED":
            if self.exists is not None or self.cycle is not None:
                raise ValueError(
                    "a budget-exceeded search returns neither claim nor witness"
                )
            return self
        _require_decided_witness_pair(self.exists, self.cycle)
        if self.cycle is not None:
            if len(self.cycle) != self.length:
                raise ValueError(
                    "witness cycle length must match the declared length"
                )
            if len(set(self.cycle)) != self.length:
                raise ValueError("witness cycle vertices must be distinct")
            if any(v >= self.graph.vertex_count for v in self.cycle):
                raise ValueError("witness cycle vertices must lie in the graph")
            edges = {
                (min(a, b), max(a, b))
                for a, b in self.graph.edges
            }
            for index, u in enumerate(self.cycle):
                v = self.cycle[(index + 1) % len(self.cycle)]
                if (min(u, v), max(u, v)) not in edges:
                    raise ValueError(
                        "witness cycle must replay as edges of the source graph"
                    )
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


class SubgraphPatternResult(SearchOutcome):
    """Whether a subgraph embedding exists, with an explicit witness.

    Carries both source graphs so the embedding replays injectivity, domain
    coverage, and exact edge preservation against the inputs.
    """

    host_graph: UndirectedGraph
    pattern_graph: UndirectedGraph
    host_vertex_count: int = Field(ge=1, le=64)
    pattern_vertex_count: int = Field(ge=1, le=64)
    exists: bool | None = None
    embedding: SubgraphEmbedding | None = None

    @model_validator(mode="after")
    def bind_witness(self) -> Self:
        if (
            self.host_vertex_count != self.host_graph.vertex_count
            or self.pattern_vertex_count != self.pattern_graph.vertex_count
        ):
            raise ValueError("vertex counts must match their source graphs")
        if self.outcome != "DECIDED":
            if self.exists is not None or self.embedding is not None:
                raise ValueError(
                    "a budget-exceeded search returns neither claim nor witness"
                )
            return self
        _require_decided_witness_pair(self.exists, self.embedding)
        if self.embedding is not None:
            mapping = dict(self.embedding.mapping)
            if set(mapping) != set(range(self.pattern_graph.vertex_count)):
                raise ValueError("embedding must cover every pattern vertex")
            if any(hv >= self.host_graph.vertex_count for hv in mapping.values()):
                raise ValueError("embedding codomain must lie in the host graph")
            host_edges = {(min(a, b), max(a, b)) for a, b in self.host_graph.edges}
            for a, b in self.pattern_graph.edges:
                ha, hb = mapping[a], mapping[b]
                if (min(ha, hb), max(ha, hb)) not in host_edges:
                    raise ValueError(
                        "embedding must preserve every pattern edge in the host"
                    )
        return self


__all__ = [
    "FixedLengthCycleRequest",
    "FixedLengthCycleResult",
    "SubgraphEmbedding",
    "SubgraphPatternRequest",
    "SubgraphPatternResult",
    "UndirectedGraph",
]
