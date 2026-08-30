from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.maximal_clique_hypergraph._models import (
    MaximalCliqueHypergraphRequest,
)
from jacobian.math.graphs.maximal_clique_hypergraph.operations import (
    construct_maximal_clique_hypergraph,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(
    vertices: Sequence[str], edges: Sequence[Sequence[str]]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((edge[0], edge[1]) for edge in edges),
    )


def test_triangle() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
    result = construct_maximal_clique_hypergraph(graph)
    assert result.clique_count == 1
    assert result.hypergraph.vertices == ("a", "b", "c")


def test_edgeless_graph() -> None:
    graph = _graph(["a", "b", "c"], [])
    result = construct_maximal_clique_hypergraph(graph)
    assert result.clique_count == 0
    assert len(result.hypergraph.edges) == 0


def test_path_graph() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"]])
    result = construct_maximal_clique_hypergraph(graph)
    assert result.clique_count == 2
    # Each edge is a maximal clique of size 2


def test_complete_graph_k4() -> None:
    graph = _graph(
        ["a", "b", "c", "d"],
        [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["b", "d"], ["c", "d"]],
    )
    result = construct_maximal_clique_hypergraph(graph)
    assert result.clique_count == 1  # The whole K4 is one maximal clique


def test_two_triangles_connected() -> None:
    graph = _graph(
        ["a", "b", "c", "d", "e"],
        [["a", "b"], ["b", "c"], ["a", "c"], ["c", "d"], ["d", "e"], ["c", "e"]],
    )
    result = construct_maximal_clique_hypergraph(graph)
    # abc is a maximal clique, cde is a maximal clique
    assert result.clique_count == 2


def test_result_preserves_source() -> None:
    graph = _graph(["a", "b"], [["a", "b"]])
    result = construct_maximal_clique_hypergraph(graph)
    assert result.graph == graph


def test_complete_nine_partite_family_is_rejected_before_enumeration() -> None:
    vertices = [f"v{part}_{slot}" for part in range(9) for slot in range(3)]
    edges = [
        [left, right]
        for left_index, left in enumerate(vertices)
        for right in vertices[left_index + 1 :]
        if left.split("_")[0] != right.split("_")[0]
    ]
    graph = _graph(vertices, edges)

    with pytest.raises(OperationDomainValidationError, match="12000-edge"):
        construct_maximal_clique_hypergraph(graph)
    with pytest.raises(ValidationError, match="12000-edge"):
        MaximalCliqueHypergraphRequest(graph=graph)
