"""Typed wire contracts for graph morphism operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.graphs.values import SimpleUndirectedGraph

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

# A negative decision costs two exhaustive passes: the operation's own search
# plus the result validator's bounded replay of that decision against the
# retained sources.  Admission charges each pass at most half the total so a
# complete negative operation stays inside the advertised budget.
SEARCH_PASSES_PER_NEGATIVE_DECISION = 2
_MAX_SEARCH_PATHS_PER_PASS = (
    MAX_CYCLE_SEARCH_PATHS // SEARCH_PASSES_PER_NEGATIVE_DECISION
)

# Results retain their source graphs, so a request near the canonical input
# limit can produce a response past the identical output limit.  Admission
# reserves this much for the result envelope beyond the echoed sources and
# witness labels.
_RESULT_ENVELOPE_RESERVE_BYTES = 1_024


def _graph_wire_bytes(graph: SimpleUndirectedGraph) -> int:
    return len(encode_strict_json(graph.model_dump(mode="json")))


def _label_wire_bytes(graph: SimpleUndirectedGraph) -> int:
    return sum(len(encode_strict_json(label) + b",") for label in graph.vertices)


def _require_output_headroom(
    source_bytes: int, witness_label_bytes: int, operation: str
) -> None:
    estimated_result_bytes = (
        source_bytes + witness_label_bytes + _RESULT_ENVELOPE_RESERVE_BYTES
    )
    output_limit = CanonicalLimits().max_output_bytes
    if estimated_result_bytes > output_limit:
        raise ValueError(
            f"the {operation} result retains its sources and would exceed the "
            f"{output_limit}-byte canonical output limit; "
            "shorten vertex labels or shrink the graphs"
        )


def _canonical_max_degree(graph: SimpleUndirectedGraph) -> int:
    degree: dict[str, int] = dict.fromkeys(graph.vertices, 0)
    for u, v in graph.edges:
        degree[u] += 1
        degree[v] += 1
    return max(degree.values(), default=0)


class FixedLengthCycleRequest(StrictModel):
    """Decide whether a simple graph contains a simple cycle of length ``k``."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Decide whether the canonical simple graph contains a simple "
                "cycle of length `length` (3..20). The request is rejected when "
                "the worst-case exhaustive search would exceed the work budget, "
                "or when the retained source graph plus witness labels would not "
                "leave enough canonical-output headroom for the echoed-source "
                "response. Accepts the domain-owned `SimpleUndirectedGraph` so "
                "callers can compose the output of `explicit_graph` or "
                "`compose_graphs` directly."
            )
        },
    )

    graph: SimpleUndirectedGraph = Field(
        description=(
            "Canonical simple undirected graph. The operation is bounded to "
            "at most 20 vertices; larger graphs are rejected."
        )
    )
    length: int = Field(ge=3, le=MORPHISM_MAX_VERTICES)

    @model_validator(mode="after")
    def require_length_within_graph(self) -> Self:
        n = len(self.graph.vertices)
        if n > MORPHISM_MAX_VERTICES:
            raise ValueError(
                f"graph must have at most {MORPHISM_MAX_VERTICES} vertices"
            )
        if self.length > n:
            raise ValueError("cycle length must not exceed the vertex count")
        # Conservative worst-case path-count bound: n * d^(length-1).
        d_max = _canonical_max_degree(self.graph)
        work = n * (d_max ** (self.length - 1))
        if work > _MAX_SEARCH_PATHS_PER_PASS:
            raise ValueError(
                "fixed-length cycle search exceeds the "
                f"{_MAX_SEARCH_PATHS_PER_PASS}-path per-pass budget "
                f"({MAX_CYCLE_SEARCH_PATHS} including validation replay)"
            )
        # The result echoes its source graph; reserve output headroom for
        # the envelope and witness labels beyond that echo.
        _require_output_headroom(
            _graph_wire_bytes(self.graph),
            _label_wire_bytes(self.graph),
            "fixed-length cycle",
        )
        return self


class FixedLengthCycleResult(StrictModel):
    """Whether a simple ``k``-cycle exists, with one ordered witness.

    The result retains its source graph so validation can replay the witness
    vertices and closing edges against it; a negative decision is accepted
    only inside the bounded request domain and only after replaying the
    exhaustive search on the retained graph. The witness vertices are
    canonical string labels from the source graph.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Cycle-existence decision bound to the source canonical graph; "
                "an EXISTS witness lists `length` distinct graph vertices "
                "(string labels) whose consecutive pairs (and the closing pair) "
                "are graph edges."
            )
        },
    )

    graph: SimpleUndirectedGraph
    decision: Literal["EXISTS", "DOES_NOT_EXIST"]
    length: int = Field(ge=3)
    cycle: tuple[str, ...] = Field(default=())

    @model_validator(mode="after")
    def require_consistent_witness(self) -> Self:  # noqa: C901
        n = len(self.graph.vertices)
        if n > 256 or len(self.graph.edges) > 32640:
            raise ValueError("source graph exceeds the canonical bounds")
        vertex_set = set(self.graph.vertices)
        # Normalise edges to unordered canonical pairs of labels.
        edge_set = set()
        for u, v in self.graph.edges:
            # SimpleUndirectedGraph already guarantees u < v and membership,
            # but re-validate for forged results.
            if u not in vertex_set or v not in vertex_set:
                raise ValueError("edge vertices must be declared")
            edge_set.add((u, v) if u < v else (v, u))
        if self.decision == "EXISTS":
            if len(self.cycle) != self.length:
                raise ValueError("an EXISTS witness must list exactly length vertices")
            if len(set(self.cycle)) != self.length:
                raise ValueError("a simple cycle witness must have distinct vertices")
            for label in self.cycle:
                if label not in vertex_set:
                    raise ValueError("cycle vertex not in source graph")
            for index in range(self.length):
                u = self.cycle[index]
                v = self.cycle[(index + 1) % self.length]
                key = (u, v) if u < v else (v, u)
                if key not in edge_set:
                    raise ValueError("a cycle witness must follow graph edges")
        else:
            if self.cycle:
                raise ValueError("a DOES_NOT_EXIST result must not carry a witness")
            # A negative conclusion is exact only inside the bounded request
            # domain; reject out-of-domain lengths BEFORE any work bound is
            # exponentiated, then mirror the admission and replay the
            # exhaustive decision against the retained graph.
            n = len(self.graph.vertices)
            if self.length > n:
                raise ValueError("cycle length must not exceed the vertex count")
            if self.length > MORPHISM_MAX_VERTICES or n > MORPHISM_MAX_VERTICES:
                raise ValueError(
                    "a DOES_NOT_EXIST decision requires the retained source "
                    f"to satisfy the {MORPHISM_MAX_VERTICES}-vertex "
                    f"{MAX_CYCLE_SEARCH_PATHS}-path request budget"
                )
            d_max = _canonical_max_degree(self.graph)
            work = n * (d_max ** (self.length - 1))
            if work > _MAX_SEARCH_PATHS_PER_PASS:
                raise ValueError(
                    "a DOES_NOT_EXIST decision requires the retained source "
                    f"to satisfy the {MORPHISM_MAX_VERTICES}-vertex "
                    f"{_MAX_SEARCH_PATHS_PER_PASS}-path request budget"
                )
            from jacobian.math.graphs.morphisms._operations import find_cycle_of_length

            if (
                find_cycle_of_length(self.graph.vertices, self.graph.edges, self.length)
                is not None
            ):
                raise ValueError(
                    "a DOES_NOT_EXIST decision contradicts the retained "
                    f"graph, which contains a cycle of length {self.length}"
                )
        return self


# ---------------------------------------------------------------------------
# Subgraph-pattern containment (non-induced subgraph monomorphism)
# ---------------------------------------------------------------------------


class SubgraphPatternFindRequest(StrictModel):
    """Find an injective edge-preserving embedding of ``pattern`` in ``host``.

    The pattern is capped at 20 vertices (``MORPHISM_MAX_VERTICES``), and the
    request is rejected when the worst-case backtracking work - bounded by the
    falling factorial of host vertices taken pattern-at-a-time times the
    edge-choice branching - would exceed the search budget. Both graphs are
    canonical ``SimpleUndirectedGraph`` values for direct composition.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Find an injective edge-preserving embedding of `pattern` in "
                "`host`. Both are canonical `SimpleUndirectedGraph` values so "
                "callers can pass `explicit_graph` output directly. `pattern` "
                "must have at most 20 vertices; requests whose worst-case "
                "assignment search exceeds the per-pass work budget are "
                "rejected. Runtime candidate scans share that budget and may "
                "return `BUDGET_EXCEEDED`; the retained sources and result "
                "envelope must fit the canonical output limit."
            )
        },
    )

    pattern: SimpleUndirectedGraph = Field(
        description="Canonical pattern graph with at most 20 vertices."
    )
    host: SimpleUndirectedGraph = Field(description="Canonical host graph.")

    @model_validator(mode="after")
    def require_search_bounded(self) -> Self:
        if len(self.pattern.vertices) > MORPHISM_MAX_VERTICES:
            raise ValueError(
                f"pattern must have at most {MORPHISM_MAX_VERTICES} vertices"
            )
        if len(self.pattern.vertices) > len(self.host.vertices):
            raise ValueError("pattern must not have more vertices than the host")
        # Conservative worst-case assignment count: P(n, k) injective maps.
        n = len(self.host.vertices)
        k = len(self.pattern.vertices)
        assignments = 1
        for step in range(k):
            assignments *= n - step
            if assignments > _MAX_SEARCH_PATHS_PER_PASS:
                raise ValueError(
                    "subgraph-pattern search exceeds the "
                    f"{_MAX_SEARCH_PATHS_PER_PASS}-assignment per-pass budget "
                    f"({MAX_CYCLE_SEARCH_PATHS} including validation replay)"
                )

        # The result echoes both source graphs; reserve output headroom for
        # the envelope and witness labels beyond those echoes.
        _require_output_headroom(
            _graph_wire_bytes(self.pattern) + _graph_wire_bytes(self.host),
            _label_wire_bytes(self.host),
            "subgraph-pattern",
        )
        return self


def _replay_subgraph_embedding(
    pattern: SimpleUndirectedGraph, host: SimpleUndirectedGraph
) -> tuple[int, ...] | None:
    """Replay a negative search with the same candidate budget as execution."""

    from jacobian.math.graphs.morphisms._operations import (
        SearchBudgetExceededError,
        find_subgraph_embedding,
    )

    try:
        return find_subgraph_embedding(
            pattern.vertices,
            pattern.edges,
            host.vertices,
            host.edges,
            max_candidate_checks=_MAX_SEARCH_PATHS_PER_PASS,
        )
    except SearchBudgetExceededError as exc:
        raise ValueError(
            "a DOES_NOT_EXIST decision exceeded the retained source "
            "candidate-check budget during validation replay"
        ) from exc


class SubgraphPatternFindResult(StrictModel):
    """Whether a non-induced subgraph embedding exists, with one witness map.

    The result retains both source graphs so validation can replay map
    length, host bounds, injectivity, and exact edge preservation; a
    negative decision is accepted only inside the bounded request domain
    and only after replaying the exhaustive containment search. The
    witness is ordered by the pattern's vertex order and contains host
    vertex labels.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Embedding decision bound to its pattern and host canonical "
                "graphs; an EXISTS vertex_map is an injective, edge-preserving "
                "map from pattern vertices (in pattern vertex order) into host "
                "vertex labels."
            )
        },
    )

    pattern: SimpleUndirectedGraph
    host: SimpleUndirectedGraph
    decision: Literal["EXISTS", "DOES_NOT_EXIST", "BUDGET_EXCEEDED"]
    vertex_map: tuple[str, ...] = Field(default=())

    @model_validator(mode="after")
    def require_consistent_pattern_witness(self) -> Self:  # noqa: C901
        if self.decision == "EXISTS":
            if len(self.vertex_map) != len(self.pattern.vertices):
                raise ValueError(
                    "an EXISTS result must map every pattern vertex exactly once"
                )
            if len(set(self.vertex_map)) != len(self.vertex_map):
                raise ValueError("a subgraph embedding must be injective")
            host_set = set(self.host.vertices)
            if any(v not in host_set for v in self.vertex_map):
                raise ValueError("vertex_map entries must be valid host vertices")
            # Host edges as unordered label pairs.
            host_edges = set()
            for u, v in self.host.edges:
                host_edges.add((u, v) if u < v else (v, u))
            # Pattern edges need mapping: pattern vertex order -> host label.
            # Build index map from pattern label -> position, then vertex_map gives host label per position.
            for u_label, v_label in self.pattern.edges:
                try:
                    u_idx = self.pattern.vertices.index(u_label)
                    v_idx = self.pattern.vertices.index(v_label)
                except ValueError:
                    raise ValueError("pattern edge vertices must be declared") from None
                mapped_u = self.vertex_map[u_idx]
                mapped_v = self.vertex_map[v_idx]
                key = (
                    (mapped_u, mapped_v)
                    if mapped_u < mapped_v
                    else (mapped_v, mapped_u)
                )
                if key not in host_edges:
                    raise ValueError("an embedding must preserve every pattern edge")
        elif self.decision == "BUDGET_EXCEEDED":
            # A budget-exhausted attempt makes no mathematical claim: it
            # must carry neither a witness nor an implicit negative.
            if self.vertex_map:
                raise ValueError("a BUDGET_EXCEEDED result must not carry a vertex map")
        else:
            if self.vertex_map:
                raise ValueError("a DOES_NOT_EXIST result must not carry a vertex map")
            # A negative conclusion is exact only inside the bounded request
            # domain; mirror that admission, then replay the exhaustive
            # containment decision against the retained graphs.
            p_n = len(self.pattern.vertices)
            h_n = len(self.host.vertices)
            assignments = 1
            for step in range(p_n):
                assignments *= h_n - step
                if assignments > _MAX_SEARCH_PATHS_PER_PASS:
                    break
            if (
                p_n > MORPHISM_MAX_VERTICES
                or p_n > h_n
                or assignments > _MAX_SEARCH_PATHS_PER_PASS
            ):
                raise ValueError(
                    "a DOES_NOT_EXIST decision requires the retained sources "
                    f"to satisfy the {MORPHISM_MAX_VERTICES}-vertex "
                    f"{_MAX_SEARCH_PATHS_PER_PASS}-assignment request budget"
                )
            found = _replay_subgraph_embedding(self.pattern, self.host)
            if found is not None:
                raise ValueError(
                    "a DOES_NOT_EXIST decision contradicts the retained "
                    "graphs, which admit an embedding"
                )
        return self
