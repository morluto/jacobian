from __future__ import annotations

from itertools import pairwise

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.path_decomposition._models import PathDecompositionResult
from jacobian.math.graphs.path_decomposition.operations import (
    compute_minimum_path_decomposition,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices, edges):
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((a, b) for a, b in edges),
    )


def test_p3_path_number_1() -> None:
    """P3 has path number 1."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    result = compute_minimum_path_decomposition(g)
    assert result.path_count == 1


def test_k3_path_number_2() -> None:
    """K3 has path number 2: one length-2 path plus one remaining edge."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")])
    result = compute_minimum_path_decomposition(g)
    assert result.path_count == 2


def test_single_edge() -> None:
    """A single edge has path number 1."""
    g = _graph(["a", "b"], [("a", "b")])
    result = compute_minimum_path_decomposition(g)
    assert result.path_count == 1


def test_edgeless_graph() -> None:
    """An edgeless graph has path number 0."""
    g = _graph(["a", "b"], [])
    result = compute_minimum_path_decomposition(g)
    assert result.path_count == 0


def test_path_replay() -> None:
    """Every source edge appears in exactly one returned path."""
    g = _graph(["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "d")])
    result = compute_minimum_path_decomposition(g)
    all_edges = set()
    for path in result.paths:
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            edge = (min(a, b), max(a, b))
            assert edge not in all_edges
            all_edges.add(edge)
    assert all_edges == set(g.edges)


def test_result_preserves_source() -> None:
    g = _graph(["a", "b"], [("a", "b")])
    result = compute_minimum_path_decomposition(g)
    assert result.graph == g


def test_k5_search_uses_memoized_residual_states() -> None:
    vertices = ["a", "b", "c", "d", "e"]
    g = _graph(
        vertices,
        [(vertices[i], vertices[j]) for i in range(5) for j in range(i + 1, 5)],
    )
    result = compute_minimum_path_decomposition(g)
    assert result.path_count > 0


def test_seven_vertex_path_is_inside_the_residual_state_bound() -> None:
    vertices = [f"v{i}" for i in range(7)]
    graph = _graph(vertices, list(pairwise(vertices)))
    result = compute_minimum_path_decomposition(graph)
    assert result.path_count == 1


def test_sparse_graph_uses_actual_path_candidates() -> None:
    """Isolated vertices do not make a one-edge search look dense."""
    vertices = [f"v{i}" for i in range(10)]
    g = _graph(vertices, [(vertices[0], vertices[1])])
    result = compute_minimum_path_decomposition(g)
    assert result.path_count == 1


def test_isolated_vertices_do_not_trigger_a_coarse_carrier_rejection() -> None:
    vertices = [f"v{i}" for i in range(13)]
    result = compute_minimum_path_decomposition(
        _graph(vertices, [(vertices[0], vertices[1])])
    )
    assert result.path_count == 1


def test_dense_graph_is_rejected_by_path_enumeration_ledger() -> None:
    vertices = [f"v{i:02d}" for i in range(12)]
    edges = [
        (vertices[left], vertices[right])
        for left in range(12)
        for right in range(left + 1, 12)
    ]

    with pytest.raises(
        OperationDomainValidationError, match=r"bounded (?:work|incidence) envelope"
    ):
        compute_minimum_path_decomposition(_graph(vertices, edges))


def test_path_orientation_is_canonical() -> None:
    graph = _graph(
        ["a", "b", "c", "d"],
        [("c", "d"), ("a", "b"), ("b", "c")],
    )

    result = compute_minimum_path_decomposition(graph)

    assert result.paths == (("a", "b", "c", "d"),)


def test_path_search_uses_reachable_residual_states() -> None:
    vertices = [f"v{i:02d}" for i in range(16)]

    result = compute_minimum_path_decomposition(
        _graph(vertices, list(pairwise(vertices)))
    )

    assert result.path_count == 1


def test_result_bound_charges_only_active_path_vertices() -> None:
    vertices = [f"{index:03d}-" + "x" * 2996 for index in range(256)]
    edges = [(vertices[2 * index], vertices[2 * index + 1]) for index in range(15)]

    result = compute_minimum_path_decomposition(_graph(vertices, edges))

    assert result.path_count == 15


@pytest.mark.parametrize(
    "payload_update",
    [
        {"path_count": -1, "paths": []},
        {"path_count": 2, "paths": [["a", "b"]]},
        {"path_count": 1, "paths": [["a", "c"]]},
        {"path_count": 1, "paths": [["a", "b", "a"]]},
    ],
)
def test_result_requires_an_exact_source_edge_partition(payload_update: dict) -> None:
    graph = _graph(["a", "b"], [("a", "b")])
    payload = {
        "graph": graph.model_dump(mode="json"),
        "path_count": 1,
        "paths": [["a", "b"]],
        **payload_update,
    }
    with pytest.raises(ValidationError):
        PathDecompositionResult.model_validate(payload)
