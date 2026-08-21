"""Typed wire contracts for graph coloring and independent set operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_EDGE_COLORING_VERTICES = 20
MAX_EDGE_COLORING_EDGES = (
    MAX_EDGE_COLORING_VERTICES * (MAX_EDGE_COLORING_VERTICES - 1) // 2
)


def _incident_edge_index_pairs_for_canonical_graph(
    graph: SimpleUndirectedGraph,
) -> list[tuple[int, int]]:
    """Return pairs of edge indices that share a vertex (must differ in color)."""
    incidence: dict[str, list[int]] = {}
    for edge_index, (u, v) in enumerate(graph.edges):
        incidence.setdefault(u, []).append(edge_index)
        incidence.setdefault(v, []).append(edge_index)
    pairs: list[tuple[int, int]] = []
    for indices in incidence.values():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                pairs.append((indices[a], indices[b]))
    return pairs


def _is_proper_edge_coloring(
    graph: SimpleUndirectedGraph,
    coloring: tuple[int, ...],
) -> bool:
    """Check whether a coloring assigns distinct colors to incident edges."""
    for a, b in _incident_edge_index_pairs_for_canonical_graph(graph):
        if coloring[a] == coloring[b]:
            return False
    return True


def _require_edge_coloring_graph_bound(graph: SimpleUndirectedGraph) -> None:
    if len(graph.vertices) > MAX_EDGE_COLORING_VERTICES:
        raise ValueError(
            f"edge-coloring supports at most {MAX_EDGE_COLORING_VERTICES} vertices"
        )


def _require_coloring_sequence(
    graph: SimpleUndirectedGraph,
    coloring: tuple[int, ...],
    colors: int,
) -> None:
    if len(coloring) != len(graph.edges):
        raise ValueError("coloring must assign one color per edge")
    for value in coloring:
        if not 0 <= value < colors:
            raise ValueError("coloring values must be in 0..colors-1")


def _require_conflicting_pair(
    graph: SimpleUndirectedGraph,
    coloring: tuple[int, ...],
    blocking_edge: tuple[str, str],
    conflicting_edge: tuple[str, str],
) -> None:
    if blocking_edge == conflicting_edge:
        raise ValueError("conflicting edge pair must be distinct")
    edge_index = {edge: idx for idx, edge in enumerate(graph.edges)}
    for edge in (blocking_edge, conflicting_edge):
        if edge[0] >= edge[1]:
            raise ValueError("blocking edges must be canonical pairs with left < right")
        if edge not in edge_index:
            raise ValueError("blocking edges must be edges of the graph")
    if not set(blocking_edge) & set(conflicting_edge):
        raise ValueError("conflicting edges must share a vertex")
    if coloring[edge_index[blocking_edge]] != coloring[edge_index[conflicting_edge]]:
        raise ValueError("conflicting edges must have the same color")


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

    graph: SimpleUndirectedGraph
    colors: StrictInt = Field(ge=1, le=20)

    @model_validator(mode="after")
    def require_bounded_graph(self) -> Self:
        _require_edge_coloring_graph_bound(self.graph)
        return self


class EdgeKColorabilityResult(StrictModel):
    """Whether a proper ``k``-edge-coloring exists, with one coloring witness."""

    graph: SimpleUndirectedGraph
    colors: StrictInt = Field(ge=1, le=20)
    colorable: bool
    coloring: tuple[StrictInt, ...] | None = Field(
        default=None, max_length=MAX_EDGE_COLORING_EDGES
    )
    edge_count: StrictInt = Field(ge=0, le=MAX_EDGE_COLORING_EDGES)

    @model_validator(mode="after")
    def require_witness_consistency(self) -> Self:
        _require_edge_coloring_graph_bound(self.graph)
        if self.edge_count != len(self.graph.edges):
            raise ValueError("edge_count must equal the number of graph edges")
        if self.colorable:
            if self.coloring is None:
                raise ValueError("a colorable result must carry a coloring witness")
            _require_coloring_sequence(self.graph, self.coloring, self.colors)
            if not _is_proper_edge_coloring(self.graph, self.coloring):
                raise ValueError("coloring witness must be a proper edge coloring")
        else:
            if self.coloring is not None:
                raise ValueError("a non-colorable result must not carry a coloring")
        return self


class EdgeColoringCheckRequest(StrictModel):
    """Validate a submitted edge-to-color assignment as a proper edge coloring."""

    graph: SimpleUndirectedGraph
    colors: StrictInt = Field(ge=1, le=20)
    coloring: tuple[StrictInt, ...] = Field(max_length=MAX_EDGE_COLORING_EDGES)

    @model_validator(mode="after")
    def require_assignment_length(self) -> Self:
        _require_edge_coloring_graph_bound(self.graph)
        _require_coloring_sequence(self.graph, self.coloring, self.colors)
        return self


class EdgeColoringCheckResult(StrictModel):
    """Whether a submitted edge coloring is proper, with a replayable conflict pair."""

    graph: SimpleUndirectedGraph
    colors: StrictInt = Field(ge=1, le=20)
    coloring: tuple[StrictInt, ...] = Field(max_length=MAX_EDGE_COLORING_EDGES)
    proper: bool
    blocking_edge: tuple[str, str] | None = None
    conflicting_edge: tuple[str, str] | None = None

    @model_validator(mode="after")
    def require_blocking_edge_consistency(self) -> Self:
        _require_edge_coloring_graph_bound(self.graph)
        _require_coloring_sequence(self.graph, self.coloring, self.colors)
        actual_proper = _is_proper_edge_coloring(self.graph, self.coloring)
        if self.proper != actual_proper:
            raise ValueError("proper flag does not match the submitted coloring")
        if self.proper:
            if self.blocking_edge is not None or self.conflicting_edge is not None:
                raise ValueError("a proper coloring must not carry a blocking edge")
            return self
        if self.blocking_edge is None or self.conflicting_edge is None:
            raise ValueError("an improper coloring must carry a conflicting edge pair")
        _require_conflicting_pair(
            self.graph, self.coloring, self.blocking_edge, self.conflicting_edge
        )
        return self
