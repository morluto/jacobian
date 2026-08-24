"""Typed wire contracts for finite hypergraph operations."""

from __future__ import annotations

import unicodedata
from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_VERTICES = 100
MAX_EDGES = 100
MAX_LABEL_LENGTH = 64


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
