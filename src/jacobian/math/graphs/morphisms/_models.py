"""Typed wire contracts for graph morphism operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

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


# ---------------------------------------------------------------------------
# Fixed-length cycle decision
# ---------------------------------------------------------------------------

# Worst-case DFS work for a k-cycle is bounded by the number of simple
# directed paths of length k-1, at most n*(d_max)^(k-1) where d_max is the
# maximum degree.  This product budget couples graph size with the requested
# length so every accepted request terminates inside a tested bound.
MAX_CYCLE_SEARCH_PATHS = 10_000_000


def _max_degree(graph: SimpleGraph) -> int:
    degree = [0] * graph.vertex_count
    for u, v in graph.edges:
        degree[u] += 1
        degree[v] += 1
    return max(degree, default=0)


class FixedLengthCycleRequest(StrictModel):
    """Decide whether a simple graph contains a simple cycle of length ``k``."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Decide whether the simple graph contains a simple cycle of "
                "length `length` (3..20). The request is rejected when the "
                "worst-case exhaustive search would exceed the work budget."
            )
        },
    )

    graph: SimpleGraph
    length: int = Field(ge=3, le=MORPHISM_MAX_VERTICES)

    @model_validator(mode="after")
    def require_length_within_graph(self) -> Self:
        if self.length > self.graph.vertex_count:
            raise ValueError("cycle length must not exceed the vertex count")
        # Conservative worst-case path-count bound: n * d^(length-1).
        d_max = _max_degree(self.graph)
        work = self.graph.vertex_count * (d_max ** (self.length - 1))
        if work > MAX_CYCLE_SEARCH_PATHS:
            raise ValueError(
                "fixed-length cycle search exceeds the "
                f"{MAX_CYCLE_SEARCH_PATHS}-path worst-case budget"
            )
        return self


class FixedLengthCycleResult(StrictModel):
    """Whether a simple ``k``-cycle exists, with one ordered witness.

    The result retains its source graph so validation can replay the witness
    vertices and closing edges against it.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Cycle-existence decision bound to the source graph; an EXISTS "
                "witness lists `length` distinct graph vertices whose "
                "consecutive pairs (and the closing pair) are graph edges."
            )
        },
    )

    graph: SimpleGraph
    decision: Literal["EXISTS", "DOES_NOT_EXIST"]
    length: int = Field(ge=3)
    cycle: tuple[int, ...] = Field(default=())

    @model_validator(mode="after")
    def require_consistent_witness(self) -> Self:
        if len(self.graph.edges) > MAX_EDGES or self.graph.vertex_count > MAX_VERTICES:
            raise ValueError("source graph exceeds the morphism bounds")
        if self.decision == "EXISTS":
            if len(self.cycle) != self.length:
                raise ValueError("an EXISTS witness must list exactly length vertices")
            if len(set(self.cycle)) != self.length:
                raise ValueError("a simple cycle witness must have distinct vertices")
            edge_set = {(min(u, v), max(u, v)) for u, v in self.graph.edges}
            for index in range(self.length):
                u = self.cycle[index]
                v = self.cycle[(index + 1) % self.length]
                if not (0 <= u < self.graph.vertex_count):
                    raise ValueError("cycle vertex out of range")
                if (min(u, v), max(u, v)) not in edge_set:
                    raise ValueError("a cycle witness must follow graph edges")
        else:
            if self.cycle:
                raise ValueError("a DOES_NOT_EXIST result must not carry a witness")
        return self


# ---------------------------------------------------------------------------
# Subgraph-pattern containment (non-induced subgraph monomorphism)
# ---------------------------------------------------------------------------


class SubgraphPatternFindRequest(StrictModel):
    """Find an injective edge-preserving embedding of ``pattern`` in ``host``.

    The pattern is capped at 20 vertices (``MORPHISM_MAX_VERTICES``), and the
    request is rejected when the worst-case backtracking work - bounded by the
    falling factorial of host vertices taken pattern-at-a-time times the
    edge-choice branching - would exceed the search budget.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Find an injective edge-preserving embedding of `pattern` in "
                "`host`. `pattern` must have at most 20 vertices; requests "
                "whose worst-case exhaustive search exceeds the work budget "
                "are rejected at validation."
            )
        },
    )

    pattern: SimpleGraph
    host: SimpleGraph

    @model_validator(mode="after")
    def require_search_bounded(self) -> Self:
        if self.pattern.vertex_count > MORPHISM_MAX_VERTICES:
            raise ValueError(
                f"pattern must have at most {MORPHISM_MAX_VERTICES} vertices"
            )
        if self.pattern.vertex_count > self.host.vertex_count:
            raise ValueError("pattern must not have more vertices than the host")
        # Conservative worst-case assignment count: P(n, k) injective maps.
        n = self.host.vertex_count
        k = self.pattern.vertex_count
        assignments = 1
        for step in range(k):
            assignments *= n - step
            if assignments > MAX_CYCLE_SEARCH_PATHS:
                raise ValueError(
                    "subgraph-pattern search exceeds the "
                    f"{MAX_CYCLE_SEARCH_PATHS}-assignment worst-case budget"
                )
        return self


class SubgraphPatternFindResult(StrictModel):
    """Whether a non-induced subgraph embedding exists, with one witness map.

    The result retains both source graphs so validation can replay map
    length, host bounds, injectivity, and exact edge preservation.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Embedding decision bound to its pattern and host graphs; an "
                "EXISTS vertex_map is an injective, edge-preserving map from "
                "pattern vertices into host vertices."
            )
        },
    )

    pattern: SimpleGraph
    host: SimpleGraph
    decision: Literal["EXISTS", "DOES_NOT_EXIST"]
    vertex_map: tuple[int, ...] = Field(default=())

    @model_validator(mode="after")
    def require_consistent_witness(self) -> Self:
        if self.decision == "EXISTS":
            if len(self.vertex_map) != self.pattern.vertex_count:
                raise ValueError(
                    "an EXISTS result must map every pattern vertex exactly once"
                )
            if len(set(self.vertex_map)) != len(self.vertex_map):
                raise ValueError("a subgraph embedding must be injective")
            if any(not 0 <= v < self.host.vertex_count for v in self.vertex_map):
                raise ValueError("vertex_map entries must be valid host vertices")
            host_edges = {(min(u, v), max(u, v)) for u, v in self.host.edges}
            for u, v in self.pattern.edges:
                mapped_u = self.vertex_map[u]
                mapped_v = self.vertex_map[v]
                if (min(mapped_u, mapped_v), max(mapped_u, mapped_v)) not in host_edges:
                    raise ValueError("an embedding must preserve every pattern edge")
        else:
            if self.vertex_map:
                raise ValueError("a DOES_NOT_EXIST result must not carry a vertex map")
        return self
