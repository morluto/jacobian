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
                0 <= source < self.vertex_count and 0 <= target < self.vertex_count
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


def _require_decided_witness_pair(exists: bool | None, witness: object) -> None:
    """A decided search states a claim; exactly an affirmative one carries a witness."""
    if exists is None:
        raise ValueError("a decided search states whether the target exists")
    if exists is (witness is None):
        raise ValueError("exactly an existing witness carries one witness")


def _require_cycle_witness_replay(
    graph: UndirectedGraph, length: int, cycle: tuple[int, ...]
) -> None:
    """A positive cycle witness must replay as edges of its source graph."""
    if len(cycle) != length:
        raise ValueError("witness cycle length must match the declared length")
    if len(set(cycle)) != length:
        raise ValueError("witness cycle vertices must be distinct")
    if any(v >= graph.vertex_count for v in cycle):
        raise ValueError("witness cycle vertices must lie in the graph")
    edges = {(min(a, b), max(a, b)) for a, b in graph.edges}
    for index, u in enumerate(cycle):
        v = cycle[(index + 1) % len(cycle)]
        if (min(u, v), max(u, v)) not in edges:
            raise ValueError("witness cycle must replay as edges of the source graph")


def _require_negative_cycle_replay(graph: UndirectedGraph, length: int) -> None:
    """Replay the same bounded exhaustive search the operation ran.

    A decided negative result implies the search completed within its
    deterministic budget, so an identical replay decides within budget as
    well; exhaustion here means the payload cannot be re-verified.
    """
    from jacobian.math.graphs.cycle_pattern._operations import (
        _BudgetExceededError,
        _decide_cycle_bounded,
    )

    try:
        contradicts = _decide_cycle_bounded(graph, length)
    except _BudgetExceededError as error:
        raise ValueError(
            "negative cycle decision cannot be re-verified within "
            "the cycle search budget"
        ) from error
    if contradicts:
        raise ValueError(
            "negative cycle decision contradicts exhaustive search: a k-cycle exists"
        )


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
        if self.graph.vertex_count != self.vertex_count:
            raise ValueError("vertex_count must match graph.vertex_count")
        if self.length > self.graph.vertex_count:
            raise ValueError("cycle length cannot exceed vertex count")
        if self.outcome != "DECIDED":
            if self.exists is not None or self.cycle is not None:
                raise ValueError(
                    "a budget-exceeded search returns neither claim nor witness"
                )
            return self
        _require_decided_witness_pair(self.exists, self.cycle)
        if self.cycle is not None:
            _require_cycle_witness_replay(self.graph, self.length, self.cycle)
        else:
            _require_negative_cycle_replay(self.graph, self.length)
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
            _require_embedding_witness_replay(
                self.host_graph, self.pattern_graph, self.embedding
            )
        else:
            _require_negative_embedding_replay(self.host_graph, self.pattern_graph)
        return self


def _require_embedding_witness_replay(
    host_graph: UndirectedGraph,
    pattern_graph: UndirectedGraph,
    embedding: SubgraphEmbedding,
) -> None:
    """A positive embedding must replay injectively and preserve edges."""
    mapping = dict(embedding.mapping)
    if set(mapping) != set(range(pattern_graph.vertex_count)):
        raise ValueError("embedding must cover every pattern vertex")
    if any(hv < 0 or hv >= host_graph.vertex_count for hv in mapping.values()):
        raise ValueError("embedding codomain must lie in the host graph")
    host_edges = {(min(a, b), max(a, b)) for a, b in host_graph.edges}
    for a, b in pattern_graph.edges:
        ha, hb = mapping[a], mapping[b]
        if (min(ha, hb), max(ha, hb)) not in host_edges:
            raise ValueError("embedding must preserve every pattern edge in the host")


def _require_negative_embedding_replay(
    host_graph: UndirectedGraph,
    pattern_graph: UndirectedGraph,
) -> None:
    """Replay the same degree-pruned bounded search the operation ran.

    A decided negative result implies the search completed within its
    deterministic budget, so an identical replay must decide within budget
    as well; exhaustion here means the payload cannot be re-verified.
    """
    from jacobian.math.graphs.cycle_pattern._operations import (
        _BudgetExceededError,
        _decide_embedding_bounded,
    )

    try:
        contradicts = _decide_embedding_bounded(host_graph, pattern_graph)
    except _BudgetExceededError as error:
        raise ValueError(
            "negative subgraph decision cannot be re-verified "
            "within the embedding search budget"
        ) from error
    if contradicts:
        raise ValueError(
            "negative subgraph decision contradicts existence of an embedding"
        )


__all__ = [
    "FixedLengthCycleRequest",
    "FixedLengthCycleResult",
    "SubgraphEmbedding",
    "SubgraphPatternRequest",
    "SubgraphPatternResult",
    "UndirectedGraph",
]
