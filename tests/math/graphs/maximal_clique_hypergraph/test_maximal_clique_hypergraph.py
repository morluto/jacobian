from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MinimumTransversalRequest,
)
from jacobian.math.graphs.maximal_clique_hypergraph._models import (
    MaximalCliqueHypergraphRequest,
    MaximalCliqueHypergraphResult,
)
from jacobian.math.graphs.maximal_clique_hypergraph._tools import (
    compute_maximal_clique_hypergraph,
)
from jacobian.math.graphs.maximal_clique_hypergraph.operations import (
    construct_maximal_clique_hypergraph,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices: list[str], edges: list[tuple[str, str]]) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((a, b) for a, b in edges),
    )


def _clique_members(
    result: MaximalCliqueHypergraphResult,
) -> set[frozenset[str]]:
    """Return the set of frozenset clique member sets."""
    return {frozenset(members) for _, members in result.hypergraph.edges}


def _independent_maximal_cliques(
    graph: SimpleUndirectedGraph,
) -> list[frozenset[str]]:
    """Independent oracle: enumerate all subsets, find maximal cliques."""
    vertices = list(graph.vertices)
    adj: dict[str, set[str]] = {v: set() for v in vertices}
    for a, b in graph.edges:
        adj[a].add(b)
        adj[b].add(a)

    all_cliques: list[frozenset[str]] = []
    for size in range(2, len(vertices) + 1):
        for subset in combinations(vertices, size):
            s = set(subset)
            if all(b in adj[a] for a, b in combinations(s, 2)):
                is_maximal = True
                for other in all_cliques:
                    if s < frozenset(other):
                        is_maximal = False
                        break
                if is_maximal:
                    all_cliques.append(frozenset(s))
    # Remove non-maximal
    final = []
    for c in all_cliques:
        is_max = True
        for other in all_cliques:
            if c != other and c < other:
                is_max = False
                break
        if is_max:
            final.append(c)
    return final


def test_edgeless_graph() -> None:
    """Edgeless graph has no nontrivial maximal cliques."""
    g = _graph(["a", "b", "c"], [])
    result = construct_maximal_clique_hypergraph(g)
    assert len(result.hypergraph.edges) == 0


def test_single_edge() -> None:
    """Single edge is the only maximal clique."""
    g = _graph(["a", "b"], [("a", "b")])
    result = construct_maximal_clique_hypergraph(g)
    assert len(result.hypergraph.edges) == 1
    members = next(iter(result.hypergraph.edges))[1]
    assert set(members) == {"a", "b"}


def test_triangle() -> None:
    """Triangle K3 has one maximal clique {a,b,c}."""
    g = _graph(
        ["a", "b", "c"],
        [("a", "b"), ("a", "c"), ("b", "c")],
    )
    result = construct_maximal_clique_hypergraph(g)
    assert len(result.hypergraph.edges) == 1
    members = next(iter(result.hypergraph.edges))[1]
    assert set(members) == {"a", "b", "c"}


def test_triangle_in_k4() -> None:
    """K4 has only one maximal clique of size 4."""
    g = _graph(
        ["a", "b", "c", "d"],
        [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")],
    )
    result = construct_maximal_clique_hypergraph(g)
    assert len(result.hypergraph.edges) == 1
    members = next(iter(result.hypergraph.edges))[1]
    assert set(members) == {"a", "b", "c", "d"}


def test_triangle_with_pendant() -> None:
    """Fixture: triangle 0-1-2 with pendant 3 attached to 2."""
    g = _graph(
        ["0", "1", "2", "3"],
        [("0", "1"), ("0", "2"), ("1", "2"), ("2", "3")],
    )
    result = construct_maximal_clique_hypergraph(g)
    cliques = _clique_members(result)
    assert frozenset({"0", "1", "2"}) in cliques
    assert frozenset({"2", "3"}) in cliques
    assert len(result.hypergraph.edges) == 2


def test_overlapping_triangles() -> None:
    """Two triangles sharing an edge produce two maximal cliques."""
    g = _graph(
        ["a", "b", "c", "d"],
        [("a", "b"), ("a", "c"), ("b", "c"), ("b", "d"), ("c", "d")],
    )
    result = construct_maximal_clique_hypergraph(g)
    assert len(result.hypergraph.edges) == 2


def test_path_graph() -> None:
    """Path a-b-c has one maximal clique {a,b,c}... no, path has no triangle."""
    g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
    result = construct_maximal_clique_hypergraph(g)
    cliques = _clique_members(result)
    assert frozenset({"a", "b"}) in cliques
    assert frozenset({"b", "c"}) in cliques
    assert len(result.hypergraph.edges) == 2


def test_cycle_graph() -> None:
    """Cycle C4 has no triangles, each edge is a maximal clique."""
    g = _graph(
        ["a", "b", "c", "d"],
        [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")],
    )
    result = construct_maximal_clique_hypergraph(g)
    assert len(result.hypergraph.edges) == 4


def test_vertex_preservation() -> None:
    """Hypergraph preserves all graph vertices."""
    g = _graph(
        ["a", "b", "c"],
        [("a", "b"), ("b", "c")],
    )
    result = construct_maximal_clique_hypergraph(g)
    assert set(result.hypergraph.vertices) == {"a", "b", "c"}


def test_exhaustive_small_comparison() -> None:
    """Compare every graph on four labelled vertices with a subset oracle."""
    vertices = ["a", "b", "c", "d"]
    candidate_edges = list(combinations(vertices, 2))
    for edge_mask in range(1 << len(candidate_edges)):
        edges = [
            edge
            for index, edge in enumerate(candidate_edges)
            if edge_mask & (1 << index)
        ]
        graph = _graph(vertices, edges)
        result = construct_maximal_clique_hypergraph(graph)
        assert _clique_members(result) == set(_independent_maximal_cliques(graph))


def test_source_retained() -> None:
    """Result retains the original graph."""
    g = _graph(["a", "b"], [("a", "b")])
    result = construct_maximal_clique_hypergraph(g)
    assert result.graph == g


def test_edge_ids_follow_source_order_deterministically() -> None:
    graph = _graph(
        ["d", "a", "c", "b"],
        [("a", "d"), ("b", "c")],
    )
    first = construct_maximal_clique_hypergraph(graph)
    second = construct_maximal_clique_hypergraph(graph)
    assert first == second
    assert first.hypergraph.edges == (
        ("clique_0", ("a", "d")),
        ("clique_1", ("b", "c")),
    )


def test_coherent_relabelling_preserves_clique_id_positions() -> None:
    original = _graph(
        ["a", "b", "c", "d"],
        [("a", "b"), ("a", "c"), ("b", "c"), ("c", "d")],
    )
    relabelled = _graph(
        ["w", "x", "y", "z"],
        [("w", "x"), ("w", "y"), ("x", "y"), ("y", "z")],
    )
    original_result = construct_maximal_clique_hypergraph(original)
    relabelled_result = construct_maximal_clique_hypergraph(relabelled)
    assert tuple(edge_id for edge_id, _ in original_result.hypergraph.edges) == tuple(
        edge_id for edge_id, _ in relabelled_result.hypergraph.edges
    )
    assert tuple(len(members) for _, members in original_result.hypergraph.edges) == (
        3,
        2,
    )
    assert tuple(len(members) for _, members in relabelled_result.hypergraph.edges) == (
        3,
        2,
    )


def test_rejects_graph_labels_outside_hypergraph_carrier() -> None:
    graph = _graph(["x" * 65], [])
    with pytest.raises(ValidationError, match="64 characters"):
        MaximalCliqueHypergraphRequest(graph=graph)
    with pytest.raises(OperationDomainValidationError, match="64 characters"):
        construct_maximal_clique_hypergraph(graph)


def test_rejects_complete_family_above_hypergraph_incidence_bound() -> None:
    parts = [[f"{part}:{offset}" for offset in range(3)] for part in range(9)]
    vertices = [vertex for part in parts for vertex in part]
    edges: list[tuple[str, str]] = [
        (left, right) if left < right else (right, left)
        for left_part, right_part in combinations(parts, 2)
        for left in left_part
        for right in right_part
    ]
    graph = _graph(vertices, edges)
    with pytest.raises(
        OperationDomainValidationError,
        match="36,000-incidence hypergraph bound",
    ):
        construct_maximal_clique_hypergraph(graph)


def test_rejects_complete_family_above_hypergraph_edge_bound() -> None:
    left = [f"left:{index}" for index in range(120)]
    right = [f"right:{index}" for index in range(101)]
    graph = _graph(
        [*left, *right],
        [(left_vertex, right_vertex) for left_vertex in left for right_vertex in right],
    )
    with pytest.raises(
        OperationDomainValidationError,
        match="12,000-edge hypergraph bound",
    ):
        construct_maximal_clique_hypergraph(graph)


def test_wire_adapter_rejects_result_beyond_transport_limit() -> None:
    left = [f"{chr(0x1D552) * 60}L{index:03}" for index in range(102)]
    right = [f"{chr(0x1D553) * 60}R{index:03}" for index in range(102)]
    graph = _graph(
        [*left, *right],
        [(left_vertex, right_vertex) for left_vertex in left for right_vertex in right],
    )
    native = construct_maximal_clique_hypergraph(graph)
    assert native.clique_count == 10_404

    with pytest.raises(
        OperationDomainValidationError,
        match="canonical output bound",
    ):
        compute_maximal_clique_hypergraph(MaximalCliqueHypergraphRequest(graph=graph))


def test_hypergraph_serializes_unchanged_into_transversal_consumer() -> None:
    graph = _graph(
        ["0", "1", "2", "3"],
        [("0", "1"), ("0", "2"), ("1", "2"), ("2", "3")],
    )
    result = construct_maximal_clique_hypergraph(graph)
    consumer = MinimumTransversalRequest.model_validate(
        {"hypergraph": result.hypergraph.model_dump(mode="json")}
    )
    assert consumer.hypergraph == result.hypergraph
