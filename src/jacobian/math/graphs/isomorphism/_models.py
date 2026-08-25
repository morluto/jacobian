"""Typed wire contracts for graph isomorphism decision operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.isomorphism._canonicalization import (
    apply_colored_graph_relabeling,
    canonicalize_colored_graph_data,
)
from jacobian.math.graphs.isomorphism._canonicalization_bounds import (
    require_admitted_colored_graph_canonicalization,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph, GraphVertexLabel


class SimpleGraph(StrictModel):
    """A simple graph (no parallel edges, no self-loops) for isomorphism.

    ``directed`` selects whether the graph is treated as directed or
    undirected.  For undirected graphs the validator canonicalises each
    edge so ``(u, v)`` and ``(v, u)`` describe the same adjacency.
    """

    vertex_count: int = Field(ge=1, le=64)
    directed: bool = False
    edges: tuple[tuple[int, int], ...] = Field(default=(), max_length=512)

    @model_validator(mode="after")
    def require_valid_edges(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for source, target in self.edges:
            if not (
                0 <= source < self.vertex_count and 0 <= target < self.vertex_count
            ):
                raise PydanticCustomError(
                    "graph.edge_vertices_must_be_in_0_vertex_count_1",
                    "edge vertices must be in 0..vertex_count-1",
                )
            if source == target:
                raise PydanticCustomError(
                    "graph.self_loops_are_not_allowed", "self-loops are not allowed"
                )
            if self.directed:
                edge_key = (source, target)
            else:
                edge_key = (min(source, target), max(source, target))
            if edge_key in seen:
                raise PydanticCustomError(
                    "graph.edges_must_be_unique", "edges must be unique"
                )
            seen.add(edge_key)
        return self


class GraphIsomorphismRequest(StrictModel):
    """Request an isomorphism decision between two simple graphs.

    The two graphs must agree on directedness and on their vertex count.
    """

    graph_a: SimpleGraph
    graph_b: SimpleGraph

    @model_validator(mode="after")
    def require_consistent_directedness(self) -> Self:
        if self.graph_a.directed != self.graph_b.directed:
            raise PydanticCustomError(
                "graph.both_graphs_must_have_the_same_directedness",
                "both graphs must have the same directedness",
            )
        return self

    @model_validator(mode="after")
    def require_consistent_vertex_count(self) -> Self:
        if self.graph_a.vertex_count != self.graph_b.vertex_count:
            raise PydanticCustomError(
                "graph.both_graphs_must_have_the_same_vertex_count",
                "both graphs must have the same vertex count",
            )
        return self


class VertexMappingPair(StrictModel):
    """One ``(from_vertex, to_vertex)`` entry in an isomorphism witness."""

    from_vertex: int = Field(ge=0, le=63)
    to_vertex: int = Field(ge=0, le=63)


class GraphIsomorphismResult(StrictModel):
    """The result of a graph isomorphism decision.

    When ``status`` is ``ISOMORPHIC`` the ``vertex_mapping`` field carries an
    explicit bijection as a list of ``(from_vertex, to_vertex)`` pairs that
    the caller can independently verify.  When ``status`` is
    ``NOT_ISOMORPHIC`` the ``vertex_mapping`` field is empty.
    """

    status: Literal["ISOMORPHIC", "NOT_ISOMORPHIC"]
    vertex_mapping: tuple[VertexMappingPair, ...] = Field(default=())
    convention: Literal["NETWORKX_IS_ISOMORPHIC"] = "NETWORKX_IS_ISOMORPHIC"


class ColoredGraphCanonicalizationRequest(StrictModel):
    """Canonicalize one materialized colored graph under color-preserving maps."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Canonicalize one materialized simple undirected graph. Vertex "
                "and edge colors are exact names: a relabeling may move vertices "
                "only when it preserves every declared color. Requests are "
                "rejected before enumeration when their color-class permutation "
                "count, execution-plus-validation replay work, or exact result "
                "size exceeds the published bound."
            )
        }
    )

    colored_graph: ColoredUndirectedGraph = Field(
        description=(
            "Canonical colored-graph value. Color tuples are empty or total and "
            "aligned with the embedded graph's authoritative vertex and edge axes."
        )
    )

    @model_validator(mode="after")
    def require_bounded_canonicalization(self) -> Self:
        require_admitted_colored_graph_canonicalization(self.colored_graph)
        return self


class GraphRelabelingPair(StrictModel):
    """One source vertex and its canonical vertex label."""

    source_vertex: GraphVertexLabel
    canonical_vertex: GraphVertexLabel


class ColoredGraphCanonicalizationResult(StrictModel):
    """A canonical colored graph and its source-bound relabeling.

    Canonical vertices are zero-padded ``v``-prefixed labels (``v00``,
    ``v01``, ...) that sort in index order up to the shared carrier's
    256-vertex bound. Vertex-color classes occupy
    positions in increasing exact color-name order; among all relabelings inside
    those classes, the canonical graph has the least sorted sequence of endpoint
    positions and edge-color names. An automorphism tie uses the least target
    tuple aligned to the source graph's vertex axis. Result validation replays
    that complete bounded search, then checks that ``relabeling`` reconstructs
    the returned graph exactly from ``source_graph``.
    """

    source_graph: ColoredUndirectedGraph
    canonical_graph: ColoredUndirectedGraph
    relabeling: tuple[GraphRelabelingPair, ...] = Field(
        max_length=256,
        description=(
            "One source-to-canonical pair per source vertex, in the source "
            "graph's authoritative vertex order."
        ),
    )

    @model_validator(mode="after")
    def require_exact_source_bound_canonical_form(self) -> Self:
        require_admitted_colored_graph_canonicalization(self.source_graph)
        if tuple(item.source_vertex for item in self.relabeling) != (
            self.source_graph.graph.vertices
        ):
            raise PydanticCustomError(
                "graph.relabeling_cover_source_vertices_their_authoritative_order",
                "relabeling must cover source vertices in their authoritative order",
            )
        mapping = {
            item.source_vertex: item.canonical_vertex for item in self.relabeling
        }
        if len(mapping) != len(self.relabeling):
            raise PydanticCustomError(
                "graph.relabeling_source_vertices_must_be_unique",
                "relabeling source vertices must be unique",
            )
        if set(mapping.values()) != set(self.canonical_graph.graph.vertices):
            raise PydanticCustomError(
                "graph.relabeling_bijection_onto_canonical_vertices",
                "relabeling must be a bijection onto the canonical graph vertices",
            )
        if apply_colored_graph_relabeling(self.source_graph, mapping) != (
            self.canonical_graph
        ):
            raise PydanticCustomError(
                "graph.relabeling_must_reconstruct_the_canonical_colore",
                "relabeling must reconstruct the canonical colored graph",
            )
        expected_graph, expected_relabeling = canonicalize_colored_graph_data(
            self.source_graph
        )
        actual_relabeling = tuple(
            (item.source_vertex, item.canonical_vertex) for item in self.relabeling
        )
        if (
            self.canonical_graph != expected_graph
            or actual_relabeling != expected_relabeling
        ):
            raise PydanticCustomError(
                "graph.result_exact_deterministic_colored_canonical_form",
                "result must be the exact deterministic colored-graph canonical form",
            )
        return self


__all__ = [
    "ColoredGraphCanonicalizationRequest",
    "ColoredGraphCanonicalizationResult",
    "GraphIsomorphismRequest",
    "GraphIsomorphismResult",
    "GraphRelabelingPair",
    "SimpleGraph",
    "VertexMappingPair",
]
