from __future__ import annotations

from jacobian.math.graphs.monochromatic_path.operations import (
    construct_monochromatic_path_hypergraphs,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph, SimpleUndirectedGraph


def _k3_red():
    return ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("0", "1", "2"),
            edges=(("0", "1"), ("0", "2"), ("1", "2")),
        ),
        edge_colors=("red", "red", "red"),
    )


def test_all_red_k3() -> None:
    """All-red K3: red path on every 2-vertex support and the full 3-vertex support."""
    result = construct_monochromatic_path_hypergraphs(_k3_red())
    red_hg = result.colour_to_hypergraph["red"]
    edges = {frozenset(m) for _, m in red_hg.edges}
    assert frozenset({"0", "1"}) in edges
    assert frozenset({"0", "2"}) in edges
    assert frozenset({"1", "2"}) in edges
    assert frozenset({"0", "1", "2"}) in edges


def test_blue_class_no_edges() -> None:
    """A blue class with no edges: only singletons."""
    g = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=("0", "1", "2"),
            edges=(("0", "1"),),
        ),
        edge_colors=("red",),
    )
    result = construct_monochromatic_path_hypergraphs(g)
    red_hg = result.colour_to_hypergraph["red"]
    red_edges = {frozenset(m) for _, m in red_hg.edges}
    assert frozenset({"0", "1"}) in red_edges
    assert frozenset({"0", "1", "2"}) not in red_edges


def test_singletons_included() -> None:
    """Singletons are included (length-0 path convention)."""
    result = construct_monochromatic_path_hypergraphs(_k3_red())
    red_hg = result.colour_to_hypergraph["red"]
    edges = {frozenset(m) for _, m in red_hg.edges}
    assert frozenset({"0"}) in edges
    assert frozenset({"1"}) in edges
    assert frozenset({"2"}) in edges


def test_replay_hamiltonian_path() -> None:
    """Each support has an actual spanning simple path in the colour subgraph."""
    result = construct_monochromatic_path_hypergraphs(_k3_red())
    red_hg = result.colour_to_hypergraph["red"]
    adjacency = {"0": {"1", "2"}, "1": {"0", "2"}, "2": {"0", "1"}}
    for _, members in red_hg.edges:
        members_list = list(members)
        if len(members_list) == 1:
            continue
        assert _has_spanning_path(members_list, adjacency)


def _has_spanning_path(vertices, adj):
    from itertools import permutations

    n = len(vertices)
    for perm in permutations(vertices):
        valid = True
        for i in range(n - 1):
            if perm[i + 1] not in adj[perm[i]]:
                valid = False
                break
        if valid:
            return True
    return False


def test_result_preserves_source() -> None:
    g = _k3_red()
    result = construct_monochromatic_path_hypergraphs(g)
    assert result.graph == g


def test_twelve_vertex_complete_graph_fits_tight_incidence_bound() -> None:
    vertices = tuple(sorted(str(index) for index in range(12)))
    edges = tuple(
        (vertices[left], vertices[right])
        for left in range(12)
        for right in range(left + 1, 12)
    )
    graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges),
        edge_colors=("red",) * len(edges),
    )

    result = construct_monochromatic_path_hypergraphs(graph)

    assert len(result.colour_to_hypergraph["red"].edges) == 2**12 - 1
