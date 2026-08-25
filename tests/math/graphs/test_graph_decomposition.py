"""Tests for structural graph decomposition operations."""

from __future__ import annotations

from typing import Literal

import networkx as nx
import pytest
from pydantic import ValidationError

from jacobian.math.graphs.decomposition._models import (
    BiconnectedComponentsRequest,
    BiconnectedComponentsResult,
    BlockCutTreeRequest,
    BlockCutTreeResult,
    BridgeBlockRequest,
    BridgeBlockResult,
    EarDecompositionRequest,
    EarDecompositionResult,
    SPQRSkeleton,
    SPQRTreeRequest,
    SPQRTreeResult,
    UndirectedGraph,
)
from jacobian.math.graphs.decomposition._operations import (
    compute_biconnected_components,
    compute_block_cut_tree,
    compute_bridge_block_tree,
    compute_ear_decomposition,
    compute_spqr_tree,
)
from jacobian.math.graphs.multigraph._models import LooplessMultigraph, MultigraphEdge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _block_cut_tree(graph: dict) -> BlockCutTreeResult:
    return compute_block_cut_tree(
        BlockCutTreeRequest.model_validate({"graph": graph}),
    )


def _bridge_block_tree(graph: dict) -> BridgeBlockResult:
    return compute_bridge_block_tree(
        BridgeBlockRequest.model_validate({"graph": graph}),
    )


def _ear_decomposition(graph: dict) -> EarDecompositionResult:
    return compute_ear_decomposition(
        EarDecompositionRequest.model_validate({"graph": graph}),
    )


def _biconnected_components(graph: dict) -> BiconnectedComponentsResult:
    return compute_biconnected_components(
        BiconnectedComponentsRequest.model_validate({"graph": graph}),
    )


def _spqr_tree(graph: dict):
    return compute_spqr_tree(SPQRTreeRequest.model_validate({"graph": graph}))


def _edges_as_sets(edges: tuple[tuple[int, int], ...]) -> frozenset:
    return frozenset((min(u, v), max(u, v)) for u, v in edges)


# ---------------------------------------------------------------------------
# UndirectedGraph validation
# ---------------------------------------------------------------------------


class TestUndirectedGraph:
    def test_valid_graph(self) -> None:
        g = UndirectedGraph(vertex_count=4, edges=((0, 1), (1, 2)))
        assert g.vertex_count == 4
        assert g.edges == ((0, 1), (1, 2))

    def test_self_loop_rejected(self) -> None:
        with pytest.raises(ValidationError):
            UndirectedGraph(vertex_count=2, edges=((0, 0),))

    def test_vertex_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            UndirectedGraph(vertex_count=2, edges=((0, 2),))

    def test_duplicate_undirected_edge_rejected(self) -> None:
        # The same edge supplied in the same orientation is a duplicate.
        with pytest.raises(ValidationError):
            UndirectedGraph(vertex_count=3, edges=((0, 1), (0, 1)))

    def test_duplicate_edge_opposite_orientation_rejected(self) -> None:
        # The same edge supplied in the opposite orientation is still a
        # duplicate for an undirected graph.
        with pytest.raises(ValidationError):
            UndirectedGraph(vertex_count=3, edges=((0, 1), (1, 0)))

    def test_vertex_count_too_large(self) -> None:
        with pytest.raises(ValidationError):
            UndirectedGraph(vertex_count=65, edges=())

    def test_vertex_count_too_small(self) -> None:
        with pytest.raises(ValidationError):
            UndirectedGraph(vertex_count=0, edges=())


# ---------------------------------------------------------------------------
# Block-cut tree
# ---------------------------------------------------------------------------


class TestBlockCutTree:
    def test_two_triangles_sharing_vertex(self) -> None:
        # Two triangles sharing vertex 0 (an articulation point).
        result = _block_cut_tree(
            {
                "vertex_count": 5,
                "edges": [
                    (0, 1),
                    (1, 2),
                    (2, 0),
                    (0, 3),
                    (3, 4),
                    (4, 0),
                ],
            },
        )
        assert len(result.blocks) == 2
        assert result.articulation_points == (0,)
        # Each block contains the articulation point.
        for block in result.blocks:
            assert 0 in block
        # The block-cut tree has one edge per (block_index, articulation_point)
        # membership.  Both blocks contain articulation point 0.
        assert (0, 0) in result.tree
        assert (1, 0) in result.tree
        assert len(result.tree) == 2

    def test_single_cycle_no_articulation(self) -> None:
        # A 4-cycle is biconnected: one block, no articulation points.
        result = _block_cut_tree(
            {
                "vertex_count": 4,
                "edges": [(0, 1), (1, 2), (2, 3), (3, 0)],
            },
        )
        assert len(result.blocks) == 1
        assert result.blocks[0] == (0, 1, 2, 3)
        assert result.articulation_points == ()
        assert result.tree == ()

    def test_path_graph(self) -> None:
        # A path of length 3: every edge is its own block (no cycles), the
        # interior vertices 1 and 2 are articulation points.
        result = _block_cut_tree(
            {
                "vertex_count": 4,
                "edges": [(0, 1), (1, 2), (2, 3)],
            },
        )
        # No cycles means each edge is a biconnected component.
        assert len(result.blocks) == 3
        # Interior vertices are articulation points.
        assert set(result.articulation_points) == {1, 2}

    def test_isolated_vertex(self) -> None:
        # An isolated vertex produces no biconnected components or
        # articulation points.
        result = _block_cut_tree(
            {"vertex_count": 2, "edges": []},
        )
        assert result.blocks == ()
        assert result.articulation_points == ()


# ---------------------------------------------------------------------------
# Bridge-block tree
# ---------------------------------------------------------------------------


class TestBridgeBlockTree:
    def test_two_triangles_with_bridge(self) -> None:
        # Two triangles joined by a single bridge edge (2, 3).
        result = _bridge_block_tree(
            {
                "vertex_count": 6,
                "edges": [
                    (0, 1),
                    (1, 2),
                    (2, 0),
                    (3, 4),
                    (4, 5),
                    (5, 3),
                    (2, 3),
                ],
            },
        )
        assert len(result.components) == 2
        assert result.bridges == ((2, 3),)
        # The bridge block tree has one edge joining the two components.
        assert len(result.tree) == 1
        u, v = result.tree[0]
        assert {u, v} == {0, 1}

    def test_cycle_no_bridges(self) -> None:
        # A 4-cycle has no bridges and forms a single 2-edge-connected
        # component.
        result = _bridge_block_tree(
            {
                "vertex_count": 4,
                "edges": [(0, 1), (1, 2), (2, 3), (3, 0)],
            },
        )
        assert len(result.components) == 1
        assert result.components[0] == (0, 1, 2, 3)
        assert result.bridges == ()
        assert result.tree == ()

    def test_path_all_bridges(self) -> None:
        # A path of length 3: every edge is a bridge, every vertex is its own
        # component.
        result = _bridge_block_tree(
            {
                "vertex_count": 4,
                "edges": [(0, 1), (1, 2), (2, 3)],
            },
        )
        # Four singleton components.
        assert len(result.components) == 4
        # Three bridges, all normalised.
        assert len(result.bridges) == 3
        for bridge in result.bridges:
            assert bridge[0] < bridge[1]
        # The tree has 3 edges (a path of 4 components).
        assert len(result.tree) == 3

    def test_bridges_are_normalised(self) -> None:
        # Bridges are returned as normalised (min, max) pairs regardless of the
        # edge orientation supplied in the input.
        result = _bridge_block_tree(
            {
                "vertex_count": 3,
                "edges": [(0, 2), (1, 0)],
            },
        )
        for bridge in result.bridges:
            assert bridge[0] < bridge[1]


# ---------------------------------------------------------------------------
# Ear decomposition
# ---------------------------------------------------------------------------


def _validate_ear_decomposition(
    ears: tuple[tuple[int, ...], ...],
    vertex_count: int,
    edges: tuple[tuple[int, int], ...],
) -> None:
    """Independently verify an ear decomposition is valid.

    A valid open ear decomposition satisfies:
    - The first ear is a cycle (its two endpoints coincide).
    - Each subsequent ear's endpoints are already used.
    - Each subsequent ear's internal vertices are new (not used before).
    - Every edge used is unused before and belongs to the input graph.
    - All input edges are eventually consumed.
    """
    assert len(ears) >= 1, "biconnected graph should have at least one ear"

    graph_edges = _edges_as_sets(edges)
    used_vertices: set[int] = set()
    used_edges: set[tuple[int, int]] = set()

    # First ear is a cycle.
    first = ears[0]
    assert first[0] == first[-1], "first ear must be a cycle"
    assert len(first) >= 3, "first ear cycle must have at least one vertex"
    used_vertices.update(first)
    for u, v in zip(first, first[1:]):  # noqa: B905, RUF007
        edge = (min(u, v), max(u, v))
        assert edge in graph_edges, f"edge {edge} not in input graph"
        assert edge not in used_edges, f"edge {edge} reused in first ear"
        used_edges.add(edge)

    # Subsequent ears.
    for ear in ears[1:]:
        assert len(ear) >= 2, f"ear {ear} too short"
        # Endpoints are used.
        assert ear[0] in used_vertices, f"ear start {ear[0]} not used"
        assert ear[-1] in used_vertices, f"ear end {ear[-1]} not used"
        # Internal vertices are new.
        for vertex in ear[1:-1]:
            assert vertex not in used_vertices, f"internal vertex {vertex} already used"
        # Edges are new and in the graph.
        for u, v in zip(ear, ear[1:]):  # noqa: B905, RUF007
            edge = (min(u, v), max(u, v))
            assert edge in graph_edges, f"edge {edge} not in input graph"
            assert edge not in used_edges, f"edge {edge} reused"
            used_edges.add(edge)
        used_vertices.update(ear)

    # All input edges are consumed.
    assert used_edges == graph_edges, (
        f"not all edges consumed: missing={graph_edges - used_edges}"
    )


class TestEarDecomposition:
    def test_cycle(self) -> None:
        edges = ((0, 1), (1, 2), (2, 3), (3, 0))
        result = _ear_decomposition(
            {
                "vertex_count": 4,
                "edges": list(edges),
            },
        )
        _validate_ear_decomposition(result.ears, 4, edges)
        # The 4-cycle is a single ear.
        assert len(result.ears) == 1

    def test_complete_graph_k4(self) -> None:
        result = _ear_decomposition(
            {
                "vertex_count": 4,
                "edges": [
                    (0, 1),
                    (0, 2),
                    (0, 3),
                    (1, 2),
                    (1, 3),
                    (2, 3),
                ],
            },
        )
        edges = (
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        )
        _validate_ear_decomposition(result.ears, 4, edges)
        # K4 has 6 edges, 4 vertices; the open ear decomposition has
        # |E| - |V| + 1 = 3 ears.
        assert len(result.ears) == 3

    def test_complete_graph_k5(self) -> None:
        edges = tuple((i, j) for i in range(5) for j in range(i))
        result = _ear_decomposition(
            {
                "vertex_count": 5,
                "edges": edges,
            },
        )
        _validate_ear_decomposition(result.ears, 5, edges)

    def test_non_biconnected_is_typed_outcome(self) -> None:
        result = _ear_decomposition(
            {
                "vertex_count": 3,
                "edges": [(0, 1), (1, 2)],
            },
        )
        assert result.biconnected is False
        assert result.ears == ()

    def test_single_vertex(self) -> None:
        # A single vertex has no ears.
        result = _ear_decomposition(
            {"vertex_count": 1, "edges": []},
        )
        assert result.ears == ()

    def test_two_vertex_edge_uses_cycle_free_convention(self) -> None:
        result = _ear_decomposition(
            {"vertex_count": 2, "edges": [(0, 1)]},
        )
        assert result.biconnected is True
        assert result.ears == ()

    def test_two_isolated_vertices_are_not_biconnected(self) -> None:
        result = _ear_decomposition(
            {"vertex_count": 2, "edges": []},
        )
        assert result.biconnected is False
        assert result.ears == ()

    def test_first_ear_is_a_cycle(self) -> None:
        result = _ear_decomposition(
            {
                "vertex_count": 3,
                "edges": [(0, 1), (1, 2), (2, 0)],
            },
        )
        first = result.ears[0]
        assert first[0] == first[-1]


# ---------------------------------------------------------------------------
# Biconnected components
# ---------------------------------------------------------------------------


class TestBiconnectedComponents:
    def test_two_triangles_sharing_vertex(self) -> None:
        result = _biconnected_components(
            {
                "vertex_count": 5,
                "edges": [
                    (0, 1),
                    (1, 2),
                    (2, 0),
                    (0, 3),
                    (3, 4),
                    (4, 0),
                ],
            },
        )
        assert len(result.components) == 2
        # Each component is a triangle.
        assert (0, 1, 2) in result.components
        assert (0, 3, 4) in result.components

    def test_single_cycle(self) -> None:
        result = _biconnected_components(
            {
                "vertex_count": 4,
                "edges": [(0, 1), (1, 2), (2, 3), (3, 0)],
            },
        )
        assert len(result.components) == 1
        assert result.components[0] == (0, 1, 2, 3)

    def test_path_graph(self) -> None:
        # A path of length 3 has no cycles; each edge is its own
        # biconnected component (a component of size 2).
        result = _biconnected_components(
            {
                "vertex_count": 4,
                "edges": [(0, 1), (1, 2), (2, 3)],
            },
        )
        assert len(result.components) == 3
        for component in result.components:
            assert len(component) == 2

    def test_isolated_vertex(self) -> None:
        # An isolated vertex produces no biconnected components.
        result = _biconnected_components(
            {"vertex_count": 2, "edges": []},
        )
        assert result.components == ()


# ---------------------------------------------------------------------------
# Integration with the operation registry
# ---------------------------------------------------------------------------


class TestOperationRegistration:
    def test_operations_registered(self) -> None:
        from jacobian.math.graphs.decomposition._tools import TOOLS

        operations = TOOLS
        operation_ids = {op.operation_id for op in operations}
        assert operation_ids == {
            "graph.decomposition.block_cut_tree.compute",
            "graph.decomposition.bridge_block_tree.compute",
            "graph.decomposition.ear.compute",
            "graph.decomposition.biconnected_components.compute",
            "graph.decomposition.spqr_tree.compute",
        }


class TestSPQRTree:
    def test_rigid_k4_is_one_r_skeleton(self) -> None:
        result = _spqr_tree(
            {
                "vertex_count": 4,
                "edges": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            }
        )
        assert result.status == "SPQR_TREE"
        assert [node.kind for node in result.nodes] == ["R_NODE"]
        assert result.virtual_edge_pairs == ()

    def test_cycle_is_one_s_skeleton(self) -> None:
        result = _spqr_tree(
            {
                "vertex_count": 4,
                "edges": [(0, 1), (1, 2), (2, 3), (3, 0)],
            }
        )
        assert result.status == "SPQR_TREE"
        assert [node.kind for node in result.nodes] == ["S_NODE"]

    def test_theta_has_parallel_junction_and_paired_virtual_edges(self) -> None:
        result = _spqr_tree(
            {
                "vertex_count": 5,
                "edges": [(0, 2), (2, 1), (0, 3), (3, 1), (0, 4), (4, 1)],
            }
        )
        assert result.status == "SPQR_TREE"
        assert [node.kind for node in result.nodes].count("P_NODE") == 1
        assert len(result.virtual_edge_pairs) == 3
        assert result.source_vertex_incidence == (
            (0, ("node:0", "node:1", "node:2", "node:3")),
            (1, ("node:0", "node:1", "node:2", "node:3")),
            (2, ("node:0",)),
            (3, ("node:1",)),
            (4, ("node:2",)),
        )
        assert len(result.source_edge_owners) == 6

    def test_non_biconnected_graph_returns_articulation_witness(self) -> None:
        result = _spqr_tree({"vertex_count": 3, "edges": [(0, 1), (1, 2)]})
        assert result.status == "NOT_BICONNECTED"
        assert result.witness_kind == "ARTICULATION"
        assert result.witness_vertices == (1,)

    def test_skeletons_reuse_the_shared_edge_id_multigraph_carrier(self) -> None:
        result = _spqr_tree(
            {
                "vertex_count": 4,
                "edges": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            }
        )
        assert isinstance(result.nodes[0].graph, LooplessMultigraph)

    def test_result_deserialization_rejects_missing_real_source_edge(self) -> None:
        result = _spqr_tree(
            {
                "vertex_count": 4,
                "edges": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            }
        )
        malformed = result.model_dump(mode="json")
        malformed["source_graph"]["edges"] = malformed["source_graph"]["edges"][:-1]
        with pytest.raises(ValueError):
            SPQRTreeResult.model_validate(malformed)

    def test_result_deserialization_rejects_virtual_pairing_and_transport_mutations(
        self,
    ) -> None:
        result = _spqr_tree(
            {
                "vertex_count": 5,
                "edges": [(0, 2), (2, 1), (0, 3), (3, 1), (0, 4), (4, 1)],
            }
        )
        malformed = result.model_dump(mode="json")
        malformed["virtual_edge_pairs"] = []
        with pytest.raises(ValueError):
            SPQRTreeResult.model_validate(malformed)
        malformed = result.model_dump(mode="json")
        malformed["source_edge_owners"][0][1] = "missing-node"
        with pytest.raises(ValueError):
            SPQRTreeResult.model_validate(malformed)
        malformed = result.model_dump(mode="json")
        malformed["source_vertex_incidence"] = []
        with pytest.raises(ValueError):
            SPQRTreeResult.model_validate(malformed)

    def test_result_deserialization_rejects_forged_branches(self) -> None:
        positive = _spqr_tree(
            {
                "vertex_count": 4,
                "edges": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            }
        ).model_dump(mode="json")
        positive["source_graph"] = {"vertex_count": 3, "edges": [[0, 1], [1, 2]]}
        with pytest.raises(ValueError):
            SPQRTreeResult.model_validate(positive)
        forged_negative = positive | {
            "source_graph": {
                "vertex_count": 4,
                "edges": [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
            },
            "status": "NOT_BICONNECTED",
            "witness_kind": "ARTICULATION",
            "witness_vertices": [0],
            "nodes": [],
            "tree_edges": [],
            "virtual_edge_pairs": [],
            "source_vertex_incidence": [],
            "source_edge_owners": [],
        }
        with pytest.raises(ValueError):
            SPQRTreeResult.model_validate(forged_negative)

    def test_result_validation_rejects_forged_empty_positive_tree(self) -> None:
        k4 = UndirectedGraph.model_validate(
            {
                "vertex_count": 4,
                "edges": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            }
        )
        with pytest.raises(ValidationError):
            SPQRTreeResult(source_graph=k4, status="SPQR_TREE")
        forged = SPQRTreeResult.model_construct(
            source_graph=k4,
            status="SPQR_TREE",
        )
        with pytest.raises(ValidationError):
            SPQRTreeResult.model_validate(forged.model_dump(mode="json"))

    def test_negative_replay_rejects_witnesses_absent_from_source(self) -> None:
        k4 = {
            "vertex_count": 4,
            "edges": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        }
        with pytest.raises(ValidationError):
            SPQRTreeResult.model_validate(
                {
                    "source_graph": k4,
                    "status": "NOT_BICONNECTED",
                    "witness_kind": "ARTICULATION",
                    "witness_vertices": [0],
                }
            )
        with pytest.raises(ValidationError):
            SPQRTreeResult.model_validate(
                {
                    "source_graph": {"vertex_count": 2, "edges": [[0, 1]]},
                    "status": "NOT_BICONNECTED",
                    "witness_kind": "DISCONNECTED",
                    "witness_vertices": [0, 99],
                }
            )
        with pytest.raises(ValidationError):
            SPQRTreeResult.model_validate(
                {
                    "source_graph": {"vertex_count": 2, "edges": [[0, 1]]},
                    "status": "NOT_BICONNECTED",
                    "witness_kind": "DISCONNECTED",
                    "witness_vertices": [0, -1],
                }
            )

    def test_replay_rejects_virtual_pair_without_genuine_separation(self) -> None:
        """A forged R+Q split of K5 must fail even though every edge maps."""
        k5_edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        genuine = _spqr_tree({"vertex_count": 5, "edges": k5_edges})
        assert [node.kind for node in genuine.nodes] == ["R_NODE"]
        rigid = genuine.nodes[0]
        moved = (3, 4)
        kept_sources = [e for e in rigid.real_edge_sources if e[1] != moved]
        kept_edges = [
            e
            for e in rigid.graph.edges
            if (rigid.vertices[e.left], rigid.vertices[e.right]) != moved
        ]
        positions = {vertex: index for index, vertex in enumerate(rigid.vertices)}
        forged_rigid = SPQRSkeleton(
            node_id=rigid.node_id,
            kind="R_NODE",
            vertices=tuple(rigid.vertices),
            graph=LooplessMultigraph(
                vertex_count=len(rigid.vertices),
                edges=(
                    *kept_edges,
                    MultigraphEdge(
                        edge_id="virtual:forged", left=positions[3], right=positions[4]
                    ),
                ),
            ),
            real_edge_sources=tuple(kept_sources),
            virtual_edge_ids=("virtual:forged",),
        )
        forged_bridge = SPQRSkeleton(
            node_id="node:forged",
            kind="Q_NODE",
            vertices=(3, 4),
            graph=LooplessMultigraph(
                vertex_count=2,
                edges=(
                    MultigraphEdge(edge_id="real:3:4", left=0, right=1),
                    MultigraphEdge(edge_id="virtual:forged-mate", left=0, right=1),
                ),
            ),
            real_edge_sources=(("real:3:4", (3, 4)),),
            virtual_edge_ids=("virtual:forged-mate",),
        )
        owners = tuple(
            sorted(
                (source, skeleton.node_id, edge_id)
                for skeleton in (forged_rigid, forged_bridge)
                for edge_id, source in skeleton.real_edge_sources
            )
        )
        incidence = tuple(
            (
                vertex,
                tuple(
                    skeleton.node_id
                    for skeleton in sorted(
                        (forged_rigid, forged_bridge), key=lambda s: s.node_id
                    )
                    if vertex in skeleton.vertices
                ),
            )
            for vertex in range(5)
        )
        with pytest.raises(ValidationError):
            SPQRTreeResult(
                source_graph=genuine.source_graph,
                status="SPQR_TREE",
                nodes=(forged_rigid, forged_bridge),
                tree_edges=(tuple(sorted(("node:forged", rigid.node_id))),),
                virtual_edge_pairs=(("virtual:forged", "virtual:forged-mate"),),
                source_vertex_incidence=incidence,
                source_edge_owners=owners,
            )

    def test_result_deserialization_rejects_extra_isolated_q_carrier(self) -> None:
        result = _spqr_tree(
            {
                "vertex_count": 4,
                "edges": [(0, 1), (1, 2), (2, 0), (0, 3), (3, 1)],
            }
        )
        assert any(node.kind == "Q_NODE" for node in result.nodes)
        malformed = result.model_dump(mode="json")
        q_node = next(node for node in malformed["nodes"] if node["kind"] == "Q_NODE")
        q_node["vertices"] = sorted([*q_node["vertices"], 9])
        q_node["graph"]["vertex_count"] = len(q_node["vertices"])
        malformed["source_vertex_incidence"] = [
            (
                vertex,
                tuple(
                    skeleton["node_id"]
                    for skeleton in sorted(
                        malformed["nodes"], key=lambda n: n["node_id"]
                    )
                    if vertex in skeleton["vertices"]
                ),
            )
            for vertex in range(malformed["source_graph"]["vertex_count"])
        ]
        with pytest.raises(ValidationError):
            SPQRTreeResult.model_validate(malformed)

    def test_replay_rejects_duplicate_virtual_pair_on_one_tree_edge(self) -> None:
        """Two K4 blocks glued along {0,1}: one tree adjacency, one gluing pair.

        The forged second gluing copies a virtual edge onto the same R
        skeleton endpoint pair, so the simpler parallel-edge invariant
        rejects it before the duplicate-tree-edge check is reached.
        """
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (0, 4),
            (0, 5),
            (1, 4),
            (1, 5),
            (4, 5),
        ]
        genuine = _spqr_tree({"vertex_count": 6, "edges": edges})
        assert genuine.status == "SPQR_TREE"
        kinds = {node.node_id: node.kind for node in genuine.nodes}
        p_id = next(node_id for node_id, kind in kinds.items() if kind == "P_NODE")
        r_id = next(node_id for node_id, kind in kinds.items() if kind == "R_NODE")
        locations = {
            edge.edge_id: node.node_id
            for node in genuine.nodes
            for edge in node.graph.edges
        }
        pair = next(
            (left, right)
            for left, right in genuine.virtual_edge_pairs
            if {kinds[locations[left]], kinds[locations[right]]} == {"P_NODE", "R_NODE"}
        )
        assert {locations[side] for side in pair} == {p_id, r_id}
        malformed = genuine.model_dump(mode="json")
        for node in malformed["nodes"]:
            if node["node_id"] not in (p_id, r_id):
                continue
            extra_id = f"virtual:forged-{node['node_id']}"
            glued = [
                edge
                for edge in node["graph"]["edges"]
                if edge["edge_id"] in (pair[0], pair[1])
            ]
            assert len(glued) == 1
            node["graph"]["edges"].append({**glued[0], "edge_id": extra_id})
            node["virtual_edge_ids"] = [*node["virtual_edge_ids"], extra_id]
        malformed["virtual_edge_pairs"] = [
            *malformed["virtual_edge_pairs"],
            [f"virtual:forged-{p_id}", f"virtual:forged-{r_id}"],
        ]
        with pytest.raises(ValidationError):
            SPQRTreeResult.model_validate(malformed)

    def test_replay_rejects_parallel_edges_inside_r_skeleton(self) -> None:
        """Two K4 blocks sharing {0,1}: a forged R/R split without the P junction."""
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (0, 4),
            (0, 5),
            (1, 4),
            (1, 5),
            (4, 5),
        ]
        genuine = _spqr_tree({"vertex_count": 6, "edges": edges})
        assert genuine.status == "SPQR_TREE"
        kinds = [node.kind for node in genuine.nodes]
        assert kinds.count("P_NODE") == 1
        assert kinds.count("Q_NODE") == 1

        def forged_rigid(
            node_id: str,
            vertices: tuple[int, ...],
            real: list[tuple[int, int]],
            virtual_id: str,
        ) -> SPQRSkeleton:
            positions = {vertex: index for index, vertex in enumerate(vertices)}
            return SPQRSkeleton(
                node_id=node_id,
                kind="R_NODE",
                vertices=vertices,
                graph=LooplessMultigraph(
                    vertex_count=len(vertices),
                    edges=(
                        *(
                            MultigraphEdge(
                                edge_id=f"real:{left}:{right}",
                                left=positions[left],
                                right=positions[right],
                            )
                            for left, right in real
                        ),
                        MultigraphEdge(
                            edge_id=virtual_id, left=positions[0], right=positions[1]
                        ),
                    ),
                ),
                real_edge_sources=tuple(
                    (f"real:{left}:{right}", (min(left, right), max(left, right)))
                    for left, right in real
                ),
                virtual_edge_ids=(virtual_id,),
            )

        shared_block = forged_rigid(
            "node:a",
            (0, 1, 2, 3),
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            "virtual:a",
        )
        other_block = forged_rigid(
            "node:b",
            (0, 1, 4, 5),
            [(0, 4), (0, 5), (1, 4), (1, 5), (4, 5)],
            "virtual:b",
        )
        shared_endpoint_pairs = [
            (shared_block.vertices[edge.left], shared_block.vertices[edge.right])
            for edge in shared_block.graph.edges
        ]
        assert len(shared_endpoint_pairs) > len(set(shared_endpoint_pairs))
        owners = tuple(
            sorted(
                (source_edge, skeleton.node_id, edge_id)
                for skeleton in (shared_block, other_block)
                for edge_id, source_edge in skeleton.real_edge_sources
            )
        )
        incidence = tuple(
            (
                vertex,
                tuple(
                    skeleton.node_id
                    for skeleton in sorted(
                        (shared_block, other_block), key=lambda s: s.node_id
                    )
                    if vertex in skeleton.vertices
                ),
            )
            for vertex in range(6)
        )
        with pytest.raises(ValidationError):
            SPQRTreeResult(
                source_graph=genuine.source_graph,
                status="SPQR_TREE",
                nodes=(shared_block, other_block),
                tree_edges=(("node:a", "node:b"),),
                virtual_edge_pairs=(("virtual:a", "virtual:b"),),
                source_vertex_incidence=incidence,
                source_edge_owners=owners,
            )

    def test_replay_rejects_disconnected_s_skeleton_of_two_cycles(self) -> None:
        """K6 plus an ear at {0,1}: a forged S skeleton of two disjoint triangles."""
        source_edges = [(i, j) for i in range(6) for j in range(i + 1, 6)]
        source_edges += [(0, 6), (1, 6)]
        genuine = _spqr_tree(
            {"vertex_count": 7, "edges": sorted(tuple(edge) for edge in source_edges)}
        )
        assert genuine.status == "SPQR_TREE"

        def forged_skeleton(
            node_id: str,
            kind: Literal["S_NODE", "R_NODE"],
            vertices: tuple[int, ...],
            real: list[tuple[int, int]],
            virtual_id: str,
        ) -> SPQRSkeleton:
            positions = {vertex: index for index, vertex in enumerate(vertices)}
            return SPQRSkeleton(
                node_id=node_id,
                kind=kind,
                vertices=vertices,
                graph=LooplessMultigraph(
                    vertex_count=len(vertices),
                    edges=(
                        *(
                            MultigraphEdge(
                                edge_id=f"real:{left}:{right}",
                                left=positions[left],
                                right=positions[right],
                            )
                            for left, right in real
                        ),
                        MultigraphEdge(
                            edge_id=virtual_id,
                            left=positions[0],
                            right=positions[1],
                        ),
                    ),
                ),
                real_edge_sources=tuple(
                    (f"real:{left}:{right}", (min(left, right), max(left, right)))
                    for left, right in real
                ),
                virtual_edge_ids=(virtual_id,),
            )

        ear_triangle = forged_skeleton(
            "node:s-forged",
            "S_NODE",
            (0, 1, 2, 3, 4, 6),
            [(0, 6), (1, 6), (2, 3), (2, 4), (3, 4)],
            "virtual:s",
        )
        remaining = [
            (a, b)
            for a, b in source_edges
            if {a, b} <= {0, 1, 2, 3, 4, 5} and (a, b) not in {(2, 3), (2, 4), (3, 4)}
        ]
        rigid = forged_skeleton(
            "node:r-forged", "R_NODE", (0, 1, 2, 3, 4, 5), remaining, "virtual:r"
        )
        assert len(ear_triangle.graph.edges) == len(ear_triangle.vertices) == 6
        owners = tuple(
            sorted(
                (source_edge, skeleton.node_id, edge_id)
                for skeleton in (ear_triangle, rigid)
                for edge_id, source_edge in skeleton.real_edge_sources
            )
        )
        incidence = tuple(
            (
                vertex,
                tuple(
                    skeleton.node_id
                    for skeleton in sorted(
                        (ear_triangle, rigid), key=lambda s: s.node_id
                    )
                    if vertex in skeleton.vertices
                ),
            )
            for vertex in range(7)
        )
        with pytest.raises(ValidationError):
            SPQRTreeResult(
                source_graph=genuine.source_graph,
                status="SPQR_TREE",
                nodes=(ear_triangle, rigid),
                tree_edges=(("node:r-forged", "node:s-forged"),),
                virtual_edge_pairs=(("virtual:s", "virtual:r"),),
                source_vertex_incidence=incidence,
                source_edge_owners=owners,
            )

    def test_request_admits_complete_graphs_on_the_declared_vertex_axis(self) -> None:
        k33_edges = [(i, j) for i in range(33) for j in range(i + 1, 33)]
        result = _spqr_tree({"vertex_count": 33, "edges": k33_edges})
        assert result.status == "SPQR_TREE"
        assert [node.kind for node in result.nodes] == ["R_NODE"]
        assert len(result.nodes[0].graph.edges) == len(k33_edges) == 528
        assert len(result.source_edge_owners) == 528
        replayed = SPQRTreeResult.model_validate(result.model_dump(mode="json"))
        assert replayed == result

    def test_result_deserialization_round_trips_genuine_decomposition(self) -> None:
        result = _spqr_tree(
            {
                "vertex_count": 5,
                "edges": [(0, 2), (2, 1), (0, 3), (3, 1), (0, 4), (4, 1)],
            }
        )
        replayed = SPQRTreeResult.model_validate(result.model_dump(mode="json"))
        assert replayed == result

    def test_replays_every_biconnected_networkx_atlas_graph(self) -> None:
        """The finite atlas covers overlapping separator patterns through 7 vertices."""
        checked = 0
        for atlas_graph in nx.graph_atlas_g():
            if not 3 <= len(atlas_graph) <= 7 or not nx.is_biconnected(atlas_graph):
                continue
            relabelled = nx.convert_node_labels_to_integers(atlas_graph)
            result = _spqr_tree(
                {
                    "vertex_count": len(relabelled),
                    "edges": tuple(sorted(relabelled.edges())),
                }
            )
            assert result.status == "SPQR_TREE"
            checked += 1
        assert checked == 538
