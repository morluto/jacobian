from __future__ import annotations

from itertools import combinations, pairwise

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.edge_deletion_profile.operations import (
    compute_edge_deletion_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices, edges):
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((a, b) for a, b in edges),
    )


def test_k3_order0() -> None:
    """K3 with deletion order 0: one row, chromatic number 3."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])
    result = compute_edge_deletion_profile(g, 0)
    assert len(result.rows) == 1
    assert result.rows[0].chromatic_number == 3
    assert result.rows[0].deleted_edge_indices == ()


def test_k3_order1() -> None:
    """K3 with deletion order 1: 4 rows (no deletion + 3 single-edge deletions)."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])
    result = compute_edge_deletion_profile(g, 1)
    assert len(result.rows) == 4  # C(3,0) + C(3,1) = 1 + 3
    # No deletion: chromatic number 3
    no_del = next(r for r in result.rows if r.deleted_edge_indices == ())
    assert no_del.chromatic_number == 3
    # Deleting one edge: chromatic number 2 (becomes a path)
    for row in result.rows:
        if row.deleted_edge_indices != ():
            assert row.chromatic_number == 2


def test_k8_order1_uses_complete_graph_shortcuts() -> None:
    vertices = [f"v{index}" for index in range(8)]
    edges = list(combinations(vertices, 2))
    result = compute_edge_deletion_profile(_graph(vertices, edges), 1)
    assert len(result.rows) == 1 + len(edges)
    assert {row.chromatic_number for row in result.rows} == {7, 8}


def test_k30_order1_uses_single_missing_edge_formula() -> None:
    vertices = [f"v{index:02d}" for index in range(30)]
    edges = list(combinations(vertices, 2))

    result = compute_edge_deletion_profile(_graph(vertices, edges), 1)

    assert len(result.rows) == 1 + len(edges)
    assert {row.chromatic_number for row in result.rows} == {29, 30}


def test_k8_order2_uses_complete_graph_shortcuts() -> None:
    vertices = [f"v{index}" for index in range(8)]
    edges = list(combinations(vertices, 2))
    result = compute_edge_deletion_profile(_graph(vertices, edges), 2)

    assert len(result.rows) == 1 + len(edges) + len(list(combinations(edges, 2)))
    assert {
        row.chromatic_number
        for row in result.rows
        if len(row.deleted_edge_indices) == 2
    } == {6, 7}


def test_large_complete_graph_with_deletions_is_rejected_for_clique_cover_work() -> None:
    vertices = [f"v{index}" for index in range(20)]
    edges = [tuple(sorted(edge)) for edge in combinations(vertices, 2)]

    with pytest.raises(OperationDomainValidationError, match="work bound"):
        compute_edge_deletion_profile(_graph(vertices, edges), 2)


def test_order2_shortcut_canonicalizes_unsorted_vertex_axis() -> None:
    vertices = ["d", "c", "b", "a"]
    edges = [tuple(sorted(edge)) for edge in combinations(vertices, 2)]
    graph = _graph(vertices, edges)
    deleted = tuple(sorted((edges.index(("a", "b")), edges.index(("a", "c")))))

    result = compute_edge_deletion_profile(graph, 2)

    row = next(row for row in result.rows if row.deleted_edge_indices == deleted)
    assert row.chromatic_number == 3


def test_near_complete_graph_order0_uses_matching_shortcut() -> None:
    vertices = [f"v{index}" for index in range(8)]
    edges = list(combinations(vertices, 2))
    edges.remove(("v0", "v1"))
    result = compute_edge_deletion_profile(_graph(vertices, edges), 0)
    assert result.rows[0].chromatic_number == 7


def test_k3_order3() -> None:
    """K3 with deletion order 3: deleting all edges yields chromatic number 1."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])
    result = compute_edge_deletion_profile(g, 3)
    # Find the row deleting all 3 edges
    all_deleted = next(r for r in result.rows if len(r.deleted_edge_indices) == 3)
    assert all_deleted.chromatic_number == 1


def test_edgeless_graph() -> None:
    """Edgeless graph with deletion order 0: chromatic number 1."""
    g = _graph(["a", "b"], [])
    result = compute_edge_deletion_profile(g, 0)
    assert len(result.rows) == 1
    assert result.rows[0].chromatic_number == 1


def test_row_count() -> None:
    """Row count = sum_{i=0}^{b} C(m, i)."""
    g = _graph(["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "d")])
    m = len(g.edges)
    for b in range(m + 1):
        result = compute_edge_deletion_profile(g, b)
        expected = sum(len(list(combinations(range(m), i))) for i in range(b + 1))
        assert len(result.rows) == expected


def test_rejects_order_exceeds_edges() -> None:
    g = _graph(["a", "b"], [("a", "b")])
    with pytest.raises(OperationDomainValidationError):
        compute_edge_deletion_profile(g, 2)


def test_admits_cheap_edgeless_graph_above_old_vertex_cap() -> None:
    graph = _graph([str(index) for index in range(9)], [])
    result = compute_edge_deletion_profile(graph, 0)
    assert len(result.rows) == 1
    assert result.rows[0].chromatic_number == 1


def test_admits_sparse_non_bipartite_component_with_isolates() -> None:
    graph = _graph(
        [f"v{index}" for index in range(256)],
        [("v0", "v1"), ("v0", "v2"), ("v1", "v2")],
    )
    result = compute_edge_deletion_profile(graph, 0)
    assert result.rows[0].chromatic_number == 3


def test_rejects_many_component_filters_before_row_enumeration() -> None:
    vertices = tuple(f"v{i}" for i in range(256))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple((f"v{2 * i}", f"v{2 * i + 1}") for i in range(128)),
    )

    with pytest.raises(OperationDomainValidationError, match="work bound"):
        compute_edge_deletion_profile(graph, 2)


def test_near_complete_large_graph_is_rejected_before_greedy_approximation() -> None:
    """A large near-complete graph cannot publish a greedy upper bound as exact."""
    vertices = [f"v{i:02}" for i in range(21)]
    missing = {tuple(sorted(pair)) for pair in pairwise(vertices[:4])}
    edges = [
        pair for pair in combinations(vertices, 2) if tuple(sorted(pair)) not in missing
    ]
    with pytest.raises(OperationDomainValidationError, match="exact work bound"):
        compute_edge_deletion_profile(
            SimpleUndirectedGraph(vertices=tuple(vertices), edges=tuple(edges)), 0
        )


def test_admits_disconnected_triangle_and_path() -> None:
    graph = _graph(
        [f"v{index:03d}" for index in range(256)],
        [("v000", "v001"), ("v000", "v002"), ("v001", "v002")]
        + [(f"v{index:03d}", f"v{index + 1:03d}") for index in range(3, 255)],
    )
    result = compute_edge_deletion_profile(graph, 0)
    assert result.rows[0].chromatic_number == 3


def test_native_negative_order_is_typed() -> None:
    graph = _graph(["a"], [])
    with pytest.raises(OperationDomainValidationError):
        compute_edge_deletion_profile(graph, -1)


def test_native_non_utf8_label_is_typed() -> None:
    graph = _graph(["\ud800"], [])
    with pytest.raises(OperationDomainValidationError):
        compute_edge_deletion_profile(graph, 0)


def test_native_oversized_label_is_rejected_before_encoding() -> None:
    graph = _graph(["x" * 11_000_000], [])
    with pytest.raises(OperationDomainValidationError, match="input/output bound"):
        compute_edge_deletion_profile(graph, 0)


def test_result_preserves_source() -> None:
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    result = compute_edge_deletion_profile(g, 1)
    assert result.graph == g
    assert result.deletion_order == 1
