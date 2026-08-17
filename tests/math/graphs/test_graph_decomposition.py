"""Tests for structural graph decomposition operations."""

from __future__ import annotations

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
    UndirectedGraph,
)
from jacobian.math.graphs.decomposition._operations import (
    compute_biconnected_components,
    compute_block_cut_tree,
    compute_bridge_block_tree,
    compute_ear_decomposition,
)

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
        }
