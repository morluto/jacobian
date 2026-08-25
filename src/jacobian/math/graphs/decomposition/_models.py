"""Typed wire contracts for structural graph decomposition operations.

All operations in this module act on an undirected simple graph supplied as
a vertex count and a tuple of ``(source, target)`` integer edges.  Vertices
are labelled ``0..vertex_count-1``; the vertex axis holds at most 64
vertices, so a simple graph admits up to ``C(64, 2) = 2016`` edges, matching
the shared multigraph carrier bounds.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.multigraph._models import MAX_EDGES, LooplessMultigraph


class UndirectedGraph(StrictModel):
    """A simple undirected graph for decomposition operations.

    The declared vertex axis bounds admission: at most 64 vertices, hence at
    most ``C(64, 2) = 2016`` distinct undirected edges.
    """

    vertex_count: int = Field(ge=1, le=64)
    edges: tuple[tuple[int, int], ...] = Field(min_length=0, max_length=MAX_EDGES)

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
            endpoint_pair = (source, target)
            canonical = (min(endpoint_pair), max(endpoint_pair))
            if canonical in seen:
                raise PydanticCustomError(
                    "graph.undirected_edges_must_be_unique",
                    "undirected edges must be unique",
                )
            seen.add(canonical)
        return self


class BlockCutTreeRequest(StrictModel):
    graph: UndirectedGraph


class BlockCutTreeResult(StrictModel):
    """Block-cut tree decomposition of a graph.

    ``blocks`` lists the biconnected components (each a sorted tuple of
    vertices), ``articulation_points`` lists the cut vertices, and ``tree``
    lists the edges of the bipartite block-cut tree joining each block to the
    articulation points it contains.
    """

    blocks: tuple[tuple[int, ...], ...] = Field(default=())
    articulation_points: tuple[int, ...] = Field(default=())
    tree: tuple[tuple[int, int], ...] = Field(default=())
    convention: Literal["NETWORKX_BICONNECTED"] = "NETWORKX_BICONNECTED"


class BridgeBlockRequest(StrictModel):
    graph: UndirectedGraph


class BridgeBlockResult(StrictModel):
    """Bridge-block (2-edge-connected component) decomposition of a graph.

    ``components`` lists each 2-edge-connected component as a sorted tuple of
    vertices, ``bridges`` lists the bridges as normalised ``(u, v)`` pairs, and
    ``tree`` lists the edges of the bridge block tree joining adjacent
    components across each bridge.
    """

    components: tuple[tuple[int, ...], ...] = Field(default=())
    bridges: tuple[tuple[int, int], ...] = Field(default=())
    tree: tuple[tuple[int, int], ...] = Field(default=())
    convention: Literal["NETWORKX_BRIDGES"] = "NETWORKX_BRIDGES"


class EarDecompositionRequest(StrictModel):
    graph: UndirectedGraph


class EarDecompositionResult(StrictModel):
    """Open ear decomposition of a biconnected graph.

    Each ear is a tuple of vertices describing a path whose internal vertex
    is disjoint from all other ears.  The first ear is a cycle.  Graphs with
    fewer than three vertices use the explicit cycle-free convention
    ``biconnected=true, ears=()``.  A graph that is not biconnected is a typed
    ``biconnected=false`` outcome.
    """

    biconnected: bool = True
    ears: tuple[tuple[int, ...], ...] = Field(default=())
    convention: Literal["JACOBIAN_EAR_DECOMPOSITION"] = "JACOBIAN_EAR_DECOMPOSITION"

    @model_validator(mode="after")
    def require_ears_match_biconnectivity(self) -> Self:
        if not self.biconnected and self.ears:
            raise PydanticCustomError(
                "graph.a_non_biconnected_graph_must_not_report_ears",
                "a non-biconnected graph must not report ears",
            )
        return self


class BiconnectedComponentsRequest(StrictModel):
    graph: UndirectedGraph


class BiconnectedComponentsResult(StrictModel):
    """All biconnected components of a graph, each a sorted tuple of vertices."""

    components: tuple[tuple[int, ...], ...] = Field(default=())
    convention: Literal["NETWORKX_BICONNECTED"] = "NETWORKX_BICONNECTED"


# ---------------------------------------------------------------------------
# SPQR trees
# ---------------------------------------------------------------------------


class SPQRTreeRequest(StrictModel):
    """Request the normalized SPQR tree of one finite simple graph.

    The positive branch uses the convention that a source graph must be
    connected, biconnected, and have at least three vertices.  Other inputs
    return a concrete ``NOT_BICONNECTED`` witness rather than an empty tree.
    Work admission follows from that convention: the split search enumerates
    at most ``C(vertex_count, 2)`` candidate separation pairs over a graph of
    at most ``C(64, 2) = 2016`` edges.
    """

    graph: UndirectedGraph = Field(
        description=(
            "Finite simple undirected source graph. A positive SPQR tree uses"
            " the biconnected, at-least-three-vertices convention."
        )
    )


class SPQRSkeleton(StrictModel):
    """One S, P, Q, or R skeleton with stable presentation identity."""

    node_id: str = Field(min_length=1, max_length=64)
    kind: Literal["S_NODE", "P_NODE", "Q_NODE", "R_NODE"]
    vertices: tuple[int, ...] = Field(min_length=2, max_length=64)
    graph: LooplessMultigraph
    real_edge_sources: tuple[tuple[str, tuple[int, int]], ...] = Field(default=())
    virtual_edge_ids: tuple[str, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical_skeleton_carrier(self) -> Self:
        if self.vertices != tuple(sorted(set(self.vertices))):
            raise PydanticCustomError(
                "graph.spqr_skeleton_vertices_must_be_sorted_and_unique",
                "SPQR skeleton vertices must be sorted and unique",
            )
        if self.graph.vertex_count != len(self.vertices):
            raise PydanticCustomError(
                "graph.spqr_multigraph_carrier_use_skeleton_vertex_axis",
                "SPQR multigraph carrier must use the skeleton vertex axis",
            )
        edge_ids = self.graph.edge_id_set
        real_ids = {edge_id for edge_id, _ in self.real_edge_sources}
        virtual_ids = set(self.virtual_edge_ids)
        if real_ids & virtual_ids or real_ids | virtual_ids != edge_ids:
            raise PydanticCustomError(
                "graph.spqr_edge_tags_must_partition_the_multigraph_edg",
                "SPQR edge tags must partition the multigraph edge IDs",
            )
        if len(real_ids) != len(self.real_edge_sources) or len(virtual_ids) != len(
            self.virtual_edge_ids
        ):
            raise PydanticCustomError(
                "graph.spqr_edge_tags_must_not_repeat_edge_ids",
                "SPQR edge tags must not repeat edge IDs",
            )
        for edge_id, source_edge in self.real_edge_sources:
            edge = self.graph.edge_by_id(edge_id)
            endpoints = (self.vertices[edge.left], self.vertices[edge.right])
            if tuple(sorted(endpoints)) != source_edge:
                raise PydanticCustomError(
                    "graph.a_real_skeleton_edge_must_name_its_source_edge",
                    "a real skeleton edge must name its source edge",
                )
        return self


class SPQRTreeResult(StrictModel):
    """Source-bound normalized SPQR decomposition or a negative witness."""

    source_graph: UndirectedGraph
    status: Literal["SPQR_TREE", "NOT_BICONNECTED"]
    witness_kind: Literal["ARTICULATION", "DISCONNECTED", "MINIMUM_SIZE"] | None = None
    witness_vertices: tuple[int, ...] = Field(default=(), max_length=2)
    nodes: tuple[SPQRSkeleton, ...] = Field(default=(), max_length=1_024)
    tree_edges: tuple[tuple[str, str], ...] = Field(default=(), max_length=1_023)
    virtual_edge_pairs: tuple[tuple[str, str], ...] = Field(
        default=(), max_length=1_536
    )
    source_vertex_incidence: tuple[tuple[int, tuple[str, ...]], ...] = Field(
        default=(), max_length=64
    )
    source_edge_owners: tuple[tuple[tuple[int, int], str, str], ...] = Field(
        default=(), max_length=MAX_EDGES
    )
    convention: Literal["JACOBIAN_NORMALIZED_FULL_SPQR_V1"] = (
        "JACOBIAN_NORMALIZED_FULL_SPQR_V1"
    )

    @model_validator(mode="after")
    def require_closed_branch_shape(self) -> Self:
        if self.status == "NOT_BICONNECTED":
            if self.witness_kind is None or not self.witness_vertices:
                raise PydanticCustomError(
                    "graph.a_non_biconnected_result_requires_a_concrete_wit",
                    "a non-biconnected result requires a concrete witness",
                )
            if (
                self.nodes
                or self.tree_edges
                or self.virtual_edge_pairs
                or self.source_vertex_incidence
                or self.source_edge_owners
            ):
                raise PydanticCustomError(
                    "graph.a_non_biconnected_result_must_not_carry_an_spqr_",
                    "a non-biconnected result must not carry an SPQR tree",
                )
        elif self.witness_kind is not None or self.witness_vertices:
            raise PydanticCustomError(
                "graph.an_spqr_tree_must_not_carry_a_negative_witness",
                "an SPQR tree must not carry a negative witness",
            )
        # Keep source-bound replay at the ordinary deserialization boundary,
        # not only in the producer. The lazy import avoids a module cycle.
        from jacobian.math.graphs.decomposition._operations import _validate_spqr_tree

        _validate_spqr_tree(self)
        return self
