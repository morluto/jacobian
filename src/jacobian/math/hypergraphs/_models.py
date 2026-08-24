"""Typed wire contracts for finite hypergraph operations."""

from __future__ import annotations

import unicodedata
from typing import Literal, Self

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    ValidationInfo,
    model_validator,
)

from jacobian._digest import Sha256Digest
from jacobian._models import StrictModel
from jacobian.canonical import encode_strict_json, sha256_digest
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_VERTICES = 100
MAX_EDGES = 100
MAX_LABEL_LENGTH = 64
MAX_SOLVER_CALLS = MAX_VERTICES

HypergraphIndependenceStatus = Literal["EXACT", "UNKNOWN"]
HypergraphIndependenceTermination = Literal[
    "OPTIMUM_ESTABLISHED",
    "WALL_TIME",
    "SOLVER_CALL_LIMIT",
    "SOLVER_ERROR",
    "SOLVER_UNKNOWN",
    "SPECIAL_CASE",
]


class FiniteHypergraph(StrictModel):
    """A finite hypergraph: a finite set of vertices and named hyperedges.

    ``vertices`` is a tuple of unique string labels.  ``edges`` is a tuple
    of ``(edge_id, vertex_subset)`` pairs where ``vertex_subset`` is a tuple
    of vertex labels.  Edge member order is irrelevant and is canonicalized
    to sorted order on construction, so two hypergraphs with the same
    members in different orders compare equal.  Every edge member must be a
    declared vertex.
    """

    vertices: tuple[str, ...] = Field(max_length=MAX_VERTICES)
    edges: tuple[tuple[str, tuple[str, ...]], ...] = Field(max_length=MAX_EDGES)

    @model_validator(mode="after")
    def require_valid_hypergraph(self) -> Self:
        labels = set(self.vertices)
        if len(labels) != len(self.vertices):
            raise ValueError("vertex labels must be distinct")
        for label in self.vertices:
            if len(label) > MAX_LABEL_LENGTH:
                raise ValueError("vertex label exceeds the bounded length budget")
        edge_ids: set[str] = set()
        canonical_edges: list[tuple[str, tuple[str, ...]]] = []
        for edge_id, members in self.edges:
            if len(edge_id) > MAX_LABEL_LENGTH:
                raise ValueError("edge id exceeds the bounded length budget")
            if edge_id in edge_ids:
                raise ValueError("edge ids must be distinct")
            edge_ids.add(edge_id)
            member_set = set(members)
            if len(member_set) != len(members):
                raise ValueError("edge members must be distinct")
            unknown = member_set - labels
            if unknown:
                raise ValueError("every edge member must be a declared vertex")
            canonical_edges.append((edge_id, tuple(sorted(members))))
        object.__setattr__(self, "edges", tuple(canonical_edges))
        return self


def _hypergraph_digest(hypergraph: FiniteHypergraph) -> str:
    payload = {
        "format": "jacobian.finite-hypergraph/v1",
        "vertices": list(hypergraph.vertices),
        "edges": [[edge_id, list(members)] for edge_id, members in hypergraph.edges],
    }
    return sha256_digest(encode_strict_json(payload))


def _greedy_independent_vertices(
    hypergraph: FiniteHypergraph,
) -> tuple[str, ...]:
    """Return one deterministic feasible incumbent in declared vertex order."""

    edge_sets = tuple(frozenset(members) for _, members in hypergraph.edges)
    selected: set[str] = set()
    for vertex in hypergraph.vertices:
        candidate = selected | {vertex}
        if not any(edge <= candidate for edge in edge_sets):
            selected.add(vertex)
    return tuple(vertex for vertex in hypergraph.vertices if vertex in selected)


def _independence_upper_bound(hypergraph: FiniteHypergraph) -> int:
    forbidden_vertices = {
        members[0] for _, members in hypergraph.edges if len(members) == 1
    }
    return len(hypergraph.vertices) - len(forbidden_vertices)


class HypergraphIndependenceBudget(StrictModel):
    """Explicit wall-time and solver-call bounds for one exact search."""

    wall_seconds: StrictInt = Field(default=5, ge=1, le=120)
    max_solver_calls: StrictInt = Field(
        default=MAX_SOLVER_CALLS,
        ge=1,
        le=MAX_SOLVER_CALLS,
        description=(
            "Maximum monotone cardinality thresholds submitted during search. "
            "Independent validation of an externally supplied result may replay "
            "one additional upper-bound threshold."
        ),
    )


class HypergraphIndependenceRequest(StrictModel):
    """One finite hypergraph and its operation-owned exact-search budget.

    Hyperedges must be nonempty. The inherited finite-hypergraph contract admits
    at most 100 vertices, 100 indexed edges, width 100, and 10,000 total
    incidences, which bounds the Boolean encoding before the private backend is
    invoked.
    """

    hypergraph: FiniteHypergraph
    resource_budget: HypergraphIndependenceBudget = Field(
        default_factory=HypergraphIndependenceBudget
    )

    @model_validator(mode="after")
    def require_supported_encoding(self) -> Self:
        if any(not members for _, members in self.hypergraph.edges):
            raise ValueError("independence-number search does not admit empty edges")
        return self


# Internal-only validation-context key. The producer-side constructor in
# ``_independence_z3`` sets it after its own threshold search has proved every
# reported bound within the same bounded call, skipping only the duplicate
# upper-bound solver replay during construction. Independently supplied results
# never carry this key and always execute the bounded replay.
_PRODUCER_ESTABLISHED_BOUNDS = "producer_established_bounds"


class HypergraphIndependenceResult(StrictModel):
    """Exact optimum or sound incumbent and bounds for one source hypergraph."""

    result_schema_version: Literal["1"] = "1"
    hypergraph: FiniteHypergraph
    hypergraph_digest: Sha256Digest
    resource_budget: HypergraphIndependenceBudget
    status: HypergraphIndependenceStatus
    independence_number: StrictInt | None = Field(default=None, ge=0, le=MAX_VERTICES)
    incumbent_vertices: tuple[str, ...] = Field(max_length=MAX_VERTICES)
    lower_bound: StrictInt = Field(ge=0, le=MAX_VERTICES)
    upper_bound: StrictInt = Field(ge=0, le=MAX_VERTICES)
    solver_calls: StrictInt = Field(ge=0, le=MAX_SOLVER_CALLS)
    wall_budget_exhausted: StrictBool
    termination_reason: HypergraphIndependenceTermination
    detail: str = Field(min_length=1, max_length=1024)
    convention: Literal["MAXIMUM_NO_COMPLETE_HYPEREDGE_VERTEX_SUBSET"] = (
        "MAXIMUM_NO_COMPLETE_HYPEREDGE_VERTEX_SUBSET"
    )

    @model_validator(mode="after")
    def bind_source_and_witness(self) -> Self:
        if any(not members for _, members in self.hypergraph.edges):
            raise ValueError("result source must not contain an empty hyperedge")
        if self.hypergraph_digest != _hypergraph_digest(self.hypergraph):
            raise ValueError("hypergraph_digest must bind the exact source hypergraph")

        witness_set = set(self.incumbent_vertices)
        expected_order = tuple(
            vertex for vertex in self.hypergraph.vertices if vertex in witness_set
        )
        if self.incumbent_vertices != expected_order or len(witness_set) != len(
            self.incumbent_vertices
        ):
            raise ValueError(
                "incumbent vertices must be unique and in declared vertex order"
            )
        if any(set(members) <= witness_set for _, members in self.hypergraph.edges):
            raise ValueError("incumbent witness must contain no complete hyperedge")
        if self.lower_bound != len(self.incumbent_vertices):
            raise ValueError("the feasible incumbent must be the lower bound")
        return self

    @model_validator(mode="after")
    def bind_bounds_to_source(self, info: ValidationInfo) -> Self:
        initial_upper = _independence_upper_bound(self.hypergraph)
        if self.solver_calls > self.resource_budget.max_solver_calls:
            raise ValueError("solver calls must fit the submitted call budget")
        if not self.lower_bound <= self.upper_bound <= initial_upper:
            raise ValueError("independence-number bounds must lie in the source range")
        producer_established = (info.context or {}).get(_PRODUCER_ESTABLISHED_BOUNDS)
        if self.upper_bound < initial_upper and not producer_established:
            from jacobian.math.hypergraphs import _independence_z3

            if not _independence_z3.verify_upper_bound(
                self.hypergraph,
                self.upper_bound,
                self.resource_budget.wall_seconds,
            ):
                raise ValueError("upper bound failed its bounded source replay")
        return self

    @model_validator(mode="after")
    def bind_exact_completion(self) -> Self:
        if self.status != "EXACT":
            return self
        initial_incumbent = _greedy_independent_vertices(self.hypergraph)
        initial_upper = _independence_upper_bound(self.hypergraph)
        if (
            self.independence_number is None
            or self.independence_number != self.lower_bound
            or self.independence_number != self.upper_bound
        ):
            raise ValueError("exact result must bind one coincident optimum")
        if self.wall_budget_exhausted:
            raise ValueError("an exact result cannot exhaust its wall budget")
        if self.termination_reason == "OPTIMUM_ESTABLISHED":
            if len(initial_incumbent) == initial_upper:
                raise ValueError("a source-trivial optimum must use SPECIAL_CASE")
            if self.independence_number < len(initial_incumbent):
                raise ValueError("an exact optimum cannot be below a feasible witness")
            expected_calls = initial_upper - self.independence_number
            if self.independence_number > len(initial_incumbent):
                expected_calls += 1
            if self.solver_calls != expected_calls:
                raise ValueError(
                    "exact solver-call count must match the descending thresholds"
                )
        elif self.termination_reason == "SPECIAL_CASE":
            if self.solver_calls != 0 or len(initial_incumbent) != initial_upper:
                raise ValueError(
                    "special-case exactness requires coincident initial bounds"
                )
        else:
            raise ValueError("exact result has an incomplete termination reason")
        return self

    @model_validator(mode="after")
    def bind_unknown_completion(self) -> Self:
        if self.status != "UNKNOWN":
            return self
        if self.independence_number is not None:
            raise ValueError("incomplete search cannot claim an independence number")
        if self.lower_bound >= self.upper_bound:
            raise ValueError("unknown result must retain a nontrivial bound gap")
        initial_upper = _independence_upper_bound(self.hypergraph)
        proved_thresholds = initial_upper - self.upper_bound
        if self.termination_reason == "SOLVER_CALL_LIMIT":
            if (
                self.wall_budget_exhausted
                or self.solver_calls != self.resource_budget.max_solver_calls
                or proved_thresholds != self.solver_calls
            ):
                raise ValueError(
                    "solver-call termination must exhaust exactly its query budget"
                )
        elif self.termination_reason == "WALL_TIME":
            if not self.wall_budget_exhausted or proved_thresholds not in {
                self.solver_calls,
                max(0, self.solver_calls - 1),
            }:
                raise ValueError(
                    "wall-time termination must bind the completed thresholds"
                )
        elif self.termination_reason == "SOLVER_UNKNOWN":
            if (
                self.wall_budget_exhausted
                or self.solver_calls == 0
                or proved_thresholds != self.solver_calls - 1
            ):
                raise ValueError(
                    "solver-unknown termination must bind its inconclusive query"
                )
        elif self.termination_reason == "SOLVER_ERROR":
            if self.wall_budget_exhausted or self.upper_bound != initial_upper:
                raise ValueError(
                    "solver-error termination must retain the source bound"
                )
        else:
            raise ValueError("unknown result has an exact termination reason")
        return self


class ParametersRequest(StrictModel):
    """Request the basic parameters of a finite hypergraph."""

    hypergraph: FiniteHypergraph


class ParametersResult(StrictModel):
    """The basic parameters of a finite hypergraph.

    ``vertex_count`` and ``edge_count`` are the number of vertices and edges.
    ``rank`` is the size of the largest edge, ``corank`` the size of the
    smallest edge, and ``uniform_size`` is that common edge size when every
    edge has the same cardinality (``None`` otherwise).
    ``total_incidences`` is the sum of all edge cardinalities.
    """

    hypergraph: FiniteHypergraph
    vertex_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    rank: int = Field(ge=0)
    corank: int = Field(ge=0)
    uniform_size: int | None = None
    total_incidences: int = Field(ge=0)

    @model_validator(mode="after")
    def bind_parameters(self) -> Self:
        from jacobian.math.hypergraphs._operations import _parameters_data

        (
            vertex_count,
            edge_count,
            rank,
            corank,
            uniform_size,
            total_incidences,
        ) = _parameters_data(self.hypergraph)
        if self.vertex_count != vertex_count:
            raise ValueError("vertex_count must be the exact number of vertices")
        if self.edge_count != edge_count:
            raise ValueError("edge_count must be the exact number of edges")
        if self.rank != rank:
            raise ValueError("rank must be the exact maximum edge size")
        if self.corank != corank:
            raise ValueError("corank must be the exact minimum edge size")
        if self.uniform_size != uniform_size:
            raise ValueError("uniform_size must match the exact uniformity")
        if self.total_incidences != total_incidences:
            raise ValueError("total_incidences must be the exact incidence count")
        return self


class VertexDegreesRequest(StrictModel):
    """Request the vertex-degree map of a finite hypergraph."""

    hypergraph: FiniteHypergraph


class VertexDegreesResult(StrictModel):
    """The vertex-degree map and degree histogram of a finite hypergraph.

    ``degrees`` maps each vertex label to its degree (the number of edges
    containing it), in declared vertex order.  ``histogram`` maps each
    degree value to the number of vertices with that degree, sorted by
    degree ascending.
    """

    hypergraph: FiniteHypergraph
    degrees: tuple[tuple[str, int], ...]
    histogram: tuple[tuple[int, int], ...]

    @model_validator(mode="after")
    def bind_vertex_degrees(self) -> Self:
        from jacobian.math.hypergraphs._operations import _vertex_degrees_data

        degrees, histogram = _vertex_degrees_data(self.hypergraph)
        if self.degrees != degrees:
            raise ValueError(
                "degrees must be the exact vertex-degree map of the hypergraph"
            )
        if self.histogram != histogram:
            raise ValueError(
                "histogram must be the exact degree histogram of the hypergraph"
            )
        return self


class DualRequest(StrictModel):
    """Request the dual of a finite hypergraph."""

    hypergraph: FiniteHypergraph


class DualResult(StrictModel):
    """The dual of a finite hypergraph.

    The dual hypergraph transposes vertices and edges: the original edges
    become vertices and the original vertices become edges, where vertex
    ``v`` becomes the edge containing one dual vertex ``e`` for each original
    edge containing ``v``.
    """

    hypergraph: FiniteHypergraph
    dual: FiniteHypergraph

    @model_validator(mode="after")
    def bind_dual(self) -> Self:
        from jacobian.math.hypergraphs._operations import _dual_data

        dual = _dual_data(self.hypergraph)
        if self.dual != dual:
            raise ValueError("dual must be the exact dual hypergraph")
        return self


class IncidenceGraphRequest(StrictModel):
    """Request the bipartite incidence graph (Levi graph) of a hypergraph."""

    hypergraph: FiniteHypergraph


class IncidenceGraphResult(StrictModel):
    """The bipartite incidence graph (Levi graph) of a finite hypergraph.

    ``vertex_incidence`` maps each vertex label to the tuple of edge ids
    containing it, in declared edge order.  ``edge_incidence`` maps each
    edge id to the tuple of vertex labels it contains, in sorted member
    order (the canonical edge order).  ``edges`` lists the
    ``(vertex, edge_id)`` incidence pairs, sorted by vertex in declared
    order then by edge id.
    """

    hypergraph: FiniteHypergraph
    vertex_incidence: tuple[tuple[str, tuple[str, ...]], ...]
    edge_incidence: tuple[tuple[str, tuple[str, ...]], ...]
    edges: tuple[tuple[str, str], ...]

    @model_validator(mode="after")
    def bind_incidence_graph(self) -> Self:
        from jacobian.math.hypergraphs._operations import _incidence_graph_data

        vertex_incidence, edge_incidence, edges = _incidence_graph_data(self.hypergraph)
        if self.vertex_incidence != vertex_incidence:
            raise ValueError("vertex_incidence must be the exact vertex-to-edges map")
        if self.edge_incidence != edge_incidence:
            raise ValueError("edge_incidence must be the exact edge-to-vertices map")
        if self.edges != edges:
            raise ValueError("edges must be the exact incidence pairs")
        return self


class CliqueExpansionRequest(StrictModel):
    """Request the clique expansion (2-section) of a hypergraph.

    The expansion is emitted as the domain-owned canonical
    ``SimpleUndirectedGraph``, whose value requires NFC-normalized vertex
    labels; hypergraphs whose declared labels are not NFC are outside this
    operation's admitted domain.
    """

    hypergraph: FiniteHypergraph

    @model_validator(mode="after")
    def require_nfc_vertex_labels(self) -> Self:
        if any(
            not unicodedata.is_normalized("NFC", vertex)
            for vertex in self.hypergraph.vertices
        ):
            raise ValueError("clique expansion requires NFC-normalized vertex labels")
        return self


class CliqueExpansionResult(StrictModel):
    """The primal/2-section graph of a finite hypergraph.

    ``graph`` is the canonical :class:`SimpleUndirectedGraph` whose vertices
    are exactly the hypergraph's declared vertex labels, in declared order,
    and in which two distinct vertices are adjacent if and only if they
    share at least one hyperedge.  Each edge's endpoints appear in lexical
    order per the canonical graph value's own convention, independent of the
    source hypergraph's declared ordering, so the value composes directly
    with downstream graph operations without translation.  This defining
    property is validated against the retained source hypergraph.
    """

    hypergraph: FiniteHypergraph
    graph: SimpleUndirectedGraph

    @model_validator(mode="after")
    def bind_clique_expansion(self) -> Self:
        from jacobian.math.hypergraphs._operations import _clique_expansion_graph

        if self.graph != _clique_expansion_graph(self.hypergraph):
            raise ValueError("graph must be the exact 2-section of the hypergraph")
        return self
