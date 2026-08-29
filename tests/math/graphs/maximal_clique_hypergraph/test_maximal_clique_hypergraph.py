from __future__ import annotations

from itertools import combinations

from jacobian.math.graphs.maximal_clique_hypergraph.operations import (
    construct_maximal_clique_hypergraph,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(vertices, edges):
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((a, b) for a, b in edges),
    )


def _clique_members(result):
    """Return the set of frozenset clique member sets."""
    return {frozenset(members) for _, members in result.hypergraph.edges}


def _independent_maximal_cliques(graph):
    """Independent oracle: enumerate all subsets, find maximal cliques."""
    vertices = list(graph.vertices)
    adj = {v: set() for v in vertices}
    for a, b in graph.edges:
        adj[a].add(b)
        adj[b].add(a)

    all_cliques = []
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
    """Compare against independent oracle for small graphs."""
    g = _graph(
        ["a", "b", "c", "d"],
        [("a", "b"), ("a", "c"), ("b", "c"), ("b", "d"), ("c", "d"), ("a", "d")],
    )
    result = construct_maximal_clique_hypergraph(g)
    expected = _independent_maximal_cliques(g)
    actual = _clique_members(result)
    assert actual == set(expected)


def test_source_retained() -> None:
    """Result retains the original graph."""
    g = _graph(["a", "b"], [("a", "b")])
    result = construct_maximal_clique_hypergraph(g)
    assert result.graph == g
