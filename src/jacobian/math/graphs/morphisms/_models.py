"""Typed wire contracts for graph morphism operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.graphs.values import (
    SimpleUndirectedGraph,
    simple_undirected_graph_wire_bytes,
)

# Exhaustive backtracking search over graph morphisms is exponential in the
# vertex count.  This dedicated bound keeps every search-based morphism
# operation inside a tested, provably bounded domain.
MORPHISM_MAX_VERTICES = 64

# Source-bound graph results retain their input graphs and may add witnesses.
# This reserved envelope covers the result wrapper and field names after those
# representation-dependent components.
_RESULT_ENVELOPE_RESERVE_BYTES = 1_024


def _label_wire_bytes(labels: tuple[str, ...]) -> int:
    return sum(len(encode_strict_json(label) + b",") for label in labels)


def _require_output_headroom(
    source_bytes: int, witness_label_bytes: int, operation: str
) -> None:
    estimated_result_bytes = (
        source_bytes + witness_label_bytes + _RESULT_ENVELOPE_RESERVE_BYTES
    )
    output_limit = CanonicalLimits().max_output_bytes
    if estimated_result_bytes > output_limit:
        raise PydanticCustomError(
            "graph.operation_result_retains_its_sources_would_exceed",
            f"the {operation} result retains its sources and would exceed the "
            f"{output_limit}-byte canonical output limit; "
            "shorten vertex labels or shrink the graphs",
        )


class GraphVertexMapRow(StrictModel):
    """One canonical source-vertex to target-vertex assignment."""

    source_vertex: str
    target_vertex: str


class GraphVertexMap(StrictModel):
    """A complete canonical vertex map between two labelled simple graphs.

    The rows are ordered by source label rather than either graph's display
    order.  This binds one total function to its precise source and target
    graphs without making row order mathematical data.
    """

    source_graph: SimpleUndirectedGraph
    target_graph: SimpleUndirectedGraph
    rows: tuple[GraphVertexMapRow, ...] = Field(max_length=256)

    @model_validator(mode="after")
    def require_complete_canonical_map(self) -> Self:
        expected_sources = tuple(sorted(self.source_graph.vertices))
        actual_sources = tuple(row.source_vertex for row in self.rows)
        if actual_sources != expected_sources:
            raise PydanticCustomError(
                "graph.vertex_map_rows_cover_every_source_vertex",
                "vertex-map rows must cover every source vertex exactly once "
                "in canonical source-label order",
            )
        target_vertices = set(self.target_graph.vertices)
        if any(row.target_vertex not in target_vertices for row in self.rows):
            raise PydanticCustomError(
                "graph.vertex_map_targets_declared_target_vertices",
                "vertex-map targets must be declared target-graph vertices",
            )

        max_obstruction_label_bytes = max(
            (
                len(encode_strict_json(label))
                for label in self.source_graph.vertices + self.target_graph.vertices
            ),
            default=0,
        )
        estimated_result_bytes = (
            len(encode_strict_json(self.model_dump(mode="json")))
            + 4 * max_obstruction_label_bytes
            + _RESULT_ENVELOPE_RESERVE_BYTES
        )
        if estimated_result_bytes > CanonicalLimits().max_output_bytes:
            raise PydanticCustomError(
                "graph.source_bound_homomorphism_result_would_exceed_canonicallimits",
                "the source-bound graph-homomorphism result would exceed the "
                f"{CanonicalLimits().max_output_bytes}-byte canonical output limit",
            )
        return self


def _first_homomorphism_obstruction(
    vertex_map: GraphVertexMap,
) -> tuple[tuple[str, str], tuple[str, str]] | None:
    """Return the lexicographically first source edge with a nonedge image."""

    images = {row.source_vertex: row.target_vertex for row in vertex_map.rows}
    target_edges = set(vertex_map.target_graph.edges)
    for source_edge in sorted(vertex_map.source_graph.edges):
        image_edge = (images[source_edge[0]], images[source_edge[1]])
        canonical_image = tuple(sorted(image_edge))
        if image_edge[0] == image_edge[1] or canonical_image not in target_edges:
            return source_edge, image_edge
    return None


class GraphHomomorphism(StrictModel):
    """A source-bound complete vertex map that preserves every source edge."""

    vertex_map: GraphVertexMap

    @model_validator(mode="after")
    def require_edge_preservation(self) -> Self:
        if _first_homomorphism_obstruction(self.vertex_map) is not None:
            raise PydanticCustomError(
                "graph.a_graph_homomorphism_must_preserve_every_source_",
                "a graph homomorphism must preserve every source edge",
            )
        return self


class GraphHomomorphismObstruction(StrictModel):
    """A source-bound first edge image that is not a target edge."""

    vertex_map: GraphVertexMap
    source_edge: tuple[str, str]
    image_vertices: tuple[str, str]

    @model_validator(mode="after")
    def require_first_replayable_obstruction(self) -> Self:
        expected = _first_homomorphism_obstruction(self.vertex_map)
        if expected is None:
            raise PydanticCustomError(
                "graph.a_homomorphism_has_no_edge_image_obstruction",
                "a homomorphism has no edge-image obstruction",
            )
        source_edge, image_vertices = expected
        if self.source_edge != source_edge:
            raise PydanticCustomError(
                "graph.obstruction_source_edge_first_failing_source_edge",
                "obstruction source_edge must be the first failing source edge",
            )
        if self.image_vertices != image_vertices:
            raise PydanticCustomError(
                "graph.obstruction_image_vertices_must_replay_the_submi",
                "obstruction image_vertices must replay the submitted map",
            )
        return self


class HomomorphismCheckRequest(StrictModel):
    """Check one complete source-bound graph vertex map exactly."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Check one complete canonical vertex map. Admission validates "
                "its retained source/map result envelope before execution; the "
                "kernel orders source edges canonically, builds target adjacency "
                "once, and scans every source edge once."
            )
        }
    )

    vertex_map: GraphVertexMap


class HomomorphismCheckResult(StrictModel):
    """A replayable positive homomorphism or first edge-image obstruction."""

    status: Literal["HOMOMORPHISM", "EDGE_IMAGE_NOT_EDGE"]
    homomorphism: GraphHomomorphism | None = None
    obstruction: GraphHomomorphismObstruction | None = None

    @model_validator(mode="after")
    def require_replayable_result(self) -> Self:
        if self.status == "HOMOMORPHISM":
            if self.obstruction is not None:
                raise PydanticCustomError(
                    "graph.a_homomorphism_result_must_not_carry_an_obstruct",
                    "a homomorphism result must not carry an obstruction",
                )
            if self.homomorphism is None:
                raise PydanticCustomError(
                    "graph.homomorphism_result_retain_checked_source_bound_map",
                    "a HOMOMORPHISM result must retain the checked source-bound map",
                )
            return self

        if self.homomorphism is not None:
            raise PydanticCustomError(
                "graph.a_non_homomorphism_result_must_not_carry_a_homom",
                "a non-homomorphism result must not carry a homomorphism",
            )
        if self.obstruction is None:
            raise PydanticCustomError(
                "graph.a_non_homomorphism_result_requires_its_first_obs",
                "a non-homomorphism result requires its first obstruction",
            )
        return self


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
            raise PydanticCustomError(
                "graph.have_at_most_morphism_max_vertices_vertices",
                f"graph must have at most {MORPHISM_MAX_VERTICES} vertices",
            )
        if self.length > n:
            raise PydanticCustomError(
                "graph.cycle_length_must_not_exceed_the_vertex_count",
                "cycle length must not exceed the vertex count",
            )
        # Conservative worst-case path-count bound: n * d^(length-1).
        d_max = _canonical_max_degree(self.graph)
        work = n * (d_max ** (self.length - 1))
        if work > _MAX_SEARCH_PATHS_PER_PASS:
            raise PydanticCustomError(
                "graph.fixed_length_cycle_search_exceeds_max_search",
                "fixed-length cycle search exceeds the "
                f"{_MAX_SEARCH_PATHS_PER_PASS}-path per-pass budget "
                f"({MAX_CYCLE_SEARCH_PATHS} including validation replay)",
            )
        # The result echoes its source graph; reserve output headroom for
        # the envelope and witness labels beyond that echo.
        _require_output_headroom(
            simple_undirected_graph_wire_bytes(self.graph),
            _label_wire_bytes(self.graph.vertices),
            "fixed-length cycle",
        )
        return self


def _cycle_source_edges(
    graph: SimpleUndirectedGraph,
) -> tuple[set[str], set[tuple[str, str]]]:
    n = len(graph.vertices)
    if n > 256 or len(graph.edges) > 32640:
        raise PydanticCustomError(
            "graph.source_graph_exceeds_the_canonical_bounds",
            "source graph exceeds the canonical bounds",
        )
    vertex_set = set(graph.vertices)
    edge_set: set[tuple[str, str]] = set()
    for u, v in graph.edges:
        if u not in vertex_set or v not in vertex_set:
            raise PydanticCustomError(
                "graph.edge_vertices_must_be_declared", "edge vertices must be declared"
            )
        edge_set.add((u, v) if u < v else (v, u))
    return vertex_set, edge_set


def _validate_cycle_witness(
    cycle: tuple[str, ...],
    length: int,
    vertex_set: set[str],
    edge_set: set[tuple[str, str]],
) -> None:
    if len(cycle) != length:
        raise PydanticCustomError(
            "graph.an_exists_witness_must_list_exactly_length_verti",
            "an EXISTS witness must list exactly length vertices",
        )
    if len(set(cycle)) != length:
        raise PydanticCustomError(
            "graph.a_simple_cycle_witness_must_have_distinct_vertic",
            "a simple cycle witness must have distinct vertices",
        )
    if any(label not in vertex_set for label in cycle):
        raise PydanticCustomError(
            "graph.cycle_vertex_not_in_source_graph", "cycle vertex not in source graph"
        )
    for index in range(length):
        u = cycle[index]
        v = cycle[(index + 1) % length]
        key = (u, v) if u < v else (v, u)
        if key not in edge_set:
            raise PydanticCustomError(
                "graph.a_cycle_witness_must_follow_graph_edges",
                "a cycle witness must follow graph edges",
            )


def _validate_negative_cycle(graph: SimpleUndirectedGraph, length: int) -> None:
    n = len(graph.vertices)
    if length > n:
        raise PydanticCustomError(
            "graph.cycle_length_must_not_exceed_the_vertex_count",
            "cycle length must not exceed the vertex count",
        )
    if length > MORPHISM_MAX_VERTICES or n > MORPHISM_MAX_VERTICES:
        raise PydanticCustomError(
            "graph.does_exist_decision_requires_retained_source_satisfy",
            "a DOES_NOT_EXIST decision requires the retained source "
            f"to satisfy the {MORPHISM_MAX_VERTICES}-vertex "
            f"{MAX_CYCLE_SEARCH_PATHS}-path request budget",
        )
    d_max = _canonical_max_degree(graph)
    work = n * (d_max ** (length - 1))
    if work > _MAX_SEARCH_PATHS_PER_PASS:
        raise PydanticCustomError(
            "graph.does_exist_decision_requires_retained_source_satisfy",
            "a DOES_NOT_EXIST decision requires the retained source "
            f"to satisfy the {MORPHISM_MAX_VERTICES}-vertex "
            f"{_MAX_SEARCH_PATHS_PER_PASS}-path request budget",
        )
    from jacobian.math.graphs.morphisms._operations import find_cycle_of_length

    if find_cycle_of_length(graph.vertices, graph.edges, length) is not None:
        raise PydanticCustomError(
            "graph.does_exist_decision_contradicts_retained_which_contains",
            "a DOES_NOT_EXIST decision contradicts the retained "
            f"graph, which contains a cycle of length {length}",
        )


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
    def require_consistent_witness(self) -> Self:
        vertex_set, edge_set = _cycle_source_edges(self.graph)
        if self.decision == "EXISTS":
            _validate_cycle_witness(self.cycle, self.length, vertex_set, edge_set)
        else:
            if self.cycle:
                raise PydanticCustomError(
                    "graph.a_does_not_exist_result_must_not_carry_a_witness",
                    "a DOES_NOT_EXIST result must not carry a witness",
                )
            _validate_negative_cycle(self.graph, self.length)
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
            raise PydanticCustomError(
                "graph.pattern_have_at_most_morphism_max_vertices",
                f"pattern must have at most {MORPHISM_MAX_VERTICES} vertices",
            )
        if len(self.pattern.vertices) > len(self.host.vertices):
            raise PydanticCustomError(
                "graph.pattern_must_not_have_more_vertices_than_the_hos",
                "pattern must not have more vertices than the host",
            )
        # Conservative worst-case assignment count: P(n, k) injective maps.
        n = len(self.host.vertices)
        k = len(self.pattern.vertices)
        assignments = 1
        for step in range(k):
            assignments *= n - step
            if assignments > _MAX_SEARCH_PATHS_PER_PASS:
                raise PydanticCustomError(
                    "graph.subgraph_pattern_search_exceeds_max_search_paths",
                    "subgraph-pattern search exceeds the "
                    f"{_MAX_SEARCH_PATHS_PER_PASS}-assignment per-pass budget "
                    f"({MAX_CYCLE_SEARCH_PATHS} including validation replay)",
                )

        # The result echoes both source graphs; reserve output headroom for
        # the envelope and witness labels beyond those echoes.
        _require_output_headroom(
            simple_undirected_graph_wire_bytes(self.pattern)
            + simple_undirected_graph_wire_bytes(self.host),
            _label_wire_bytes(self.host.vertices),
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
        raise PydanticCustomError(
            "graph.does_exist_decision_exceeded_retained_source_candidate",
            "a DOES_NOT_EXIST decision exceeded the retained source "
            "candidate-check budget during validation replay",
        ) from exc


def _validate_embedding_witness(
    pattern: SimpleUndirectedGraph,
    host: SimpleUndirectedGraph,
    vertex_map: tuple[str, ...],
) -> None:
    if len(vertex_map) != len(pattern.vertices):
        raise PydanticCustomError(
            "graph.an_exists_result_must_map_every_pattern_vertex_e",
            "an EXISTS result must map every pattern vertex exactly once",
        )
    if len(set(vertex_map)) != len(vertex_map):
        raise PydanticCustomError(
            "graph.a_subgraph_embedding_must_be_injective",
            "a subgraph embedding must be injective",
        )
    host_set = set(host.vertices)
    if any(v not in host_set for v in vertex_map):
        raise PydanticCustomError(
            "graph.vertex_map_entries_must_be_valid_host_vertices",
            "vertex_map entries must be valid host vertices",
        )
    host_edges = {(u, v) if u < v else (v, u) for u, v in host.edges}
    for u_label, v_label in pattern.edges:
        try:
            u_idx = pattern.vertices.index(u_label)
            v_idx = pattern.vertices.index(v_label)
        except ValueError:
            raise PydanticCustomError(
                "graph.pattern_edge_vertices_must_be_declared",
                "pattern edge vertices must be declared",
            ) from None
        mapped_u = vertex_map[u_idx]
        mapped_v = vertex_map[v_idx]
        key = (mapped_u, mapped_v) if mapped_u < mapped_v else (mapped_v, mapped_u)
        if key not in host_edges:
            raise PydanticCustomError(
                "graph.an_embedding_must_preserve_every_pattern_edge",
                "an embedding must preserve every pattern edge",
            )


def _validate_negative_embedding(
    pattern: SimpleUndirectedGraph,
    host: SimpleUndirectedGraph,
) -> None:
    p_n = len(pattern.vertices)
    h_n = len(host.vertices)
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
        raise PydanticCustomError(
            "graph.does_exist_decision_requires_retained_sources_satisfy",
            "a DOES_NOT_EXIST decision requires the retained sources "
            f"to satisfy the {MORPHISM_MAX_VERTICES}-vertex "
            f"{_MAX_SEARCH_PATHS_PER_PASS}-assignment request budget",
        )
    if _replay_subgraph_embedding(pattern, host) is not None:
        raise PydanticCustomError(
            "graph.does_exist_decision_contradicts_retained_graphs_which",
            "a DOES_NOT_EXIST decision contradicts the retained "
            "graphs, which admit an embedding",
        )


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
    def require_consistent_pattern_witness(self) -> Self:
        if self.decision == "EXISTS":
            _validate_embedding_witness(self.pattern, self.host, self.vertex_map)
        elif self.decision == "BUDGET_EXCEEDED":
            # A budget-exhausted attempt makes no mathematical claim: it
            # must carry neither a witness nor an implicit negative.
            if self.vertex_map:
                raise PydanticCustomError(
                    "graph.a_budget_exceeded_result_must_not_carry_a_vertex",
                    "a BUDGET_EXCEEDED result must not carry a vertex map",
                )
        else:
            if self.vertex_map:
                raise PydanticCustomError(
                    "graph.a_does_not_exist_result_must_not_carry_a_vertex_",
                    "a DOES_NOT_EXIST result must not carry a vertex map",
                )
            _validate_negative_embedding(self.pattern, self.host)
        return self
