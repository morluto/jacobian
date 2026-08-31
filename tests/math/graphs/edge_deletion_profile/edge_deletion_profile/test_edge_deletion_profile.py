from __future__ import annotations

from collections.abc import Sequence

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.edge_deletion_profile._models import (
    EdgeDeletionProfileRequest,
)
from jacobian.math.graphs.edge_deletion_profile.operations import (
    compute_edge_deletion_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(
    vertices: Sequence[str], edges: Sequence[Sequence[str]]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((edge[0], edge[1]) for edge in edges),
    )


def test_triangle_d0() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
    result = compute_edge_deletion_profile(graph, 0)
    assert len(result.rows) == 1
    assert result.rows[0].chromatic_number == 3


def test_triangle_d1() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
    result = compute_edge_deletion_profile(graph, 1)
    assert len(result.rows) == 4
    for row in result.rows[1:]:
        assert row.chromatic_number == 2


def test_edgeless_graph() -> None:
    graph = _graph(["a", "b", "c"], [])
    result = compute_edge_deletion_profile(graph, 0)
    assert len(result.rows) == 1
    assert result.rows[0].chromatic_number == 1


def test_null_graph_has_chromatic_number_zero() -> None:
    result = compute_edge_deletion_profile(_graph([], []), 0)
    assert result.rows[0].chromatic_number == 0


def test_path_d0() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"]])
    result = compute_edge_deletion_profile(graph, 0)
    assert result.rows[0].chromatic_number == 2


def test_path_d1() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"]])
    result = compute_edge_deletion_profile(graph, 1)
    assert len(result.rows) == 3
    # After deleting any edge from a path of 3, we get an edge + isolated vertex
    for row in result.rows[1:]:
        assert row.chromatic_number == 2


def test_result_preserves_source() -> None:
    graph = _graph(["a", "b"], [["a", "b"]])
    result = compute_edge_deletion_profile(graph, 0)
    assert result.graph == graph


def test_deleting_every_bipartite_edge_uses_exact_fast_path() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"]])

    result = compute_edge_deletion_profile(graph, 2)

    assert result.rows[-1].deleted_edge_indices == (0, 1)
    assert result.rows[-1].chromatic_number == 1


def test_large_subset_family_is_rejected_before_expansion() -> None:
    vertices = [f"v{i:02d}" for i in range(80)]
    edges = [[vertices[2 * index], vertices[2 * index + 1]] for index in range(40)]
    graph = _graph(vertices, edges)

    with pytest.raises(OperationDomainValidationError, match="work bound"):
        compute_edge_deletion_profile(graph, 20)
    request = EdgeDeletionProfileRequest(graph=graph, deletion_order=20)
    assert request.deletion_order == 20


def test_negative_deletion_order_is_rejected() -> None:
    graph = _graph(["a", "b"], [["a", "b"]])
    with pytest.raises(OperationDomainValidationError, match="integer between"):
        compute_edge_deletion_profile(graph, -1)
