"""Independent mathematical tests for exact spanning-tree counts."""

from itertools import combinations

from jacobian.math.graphs.optimization._chromatic_kernel import build_simple_graph
from jacobian.math.graphs.optimization._invariants import _spanning_tree_count
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _brute_force_count(graph: SimpleUndirectedGraph) -> int:
    order = len(graph.vertices)
    if order == 0:
        return 0
    if order == 1:
        return 1
    edges = tuple(graph.edges)
    count = 0
    for selected in combinations(edges, order - 1):
        parent = {vertex: vertex for vertex in graph.vertices}

        def root(vertex: str, parents: dict[str, str] = parent) -> str:
            while parents[vertex] != vertex:
                parents[vertex] = parents[parents[vertex]]
                vertex = parents[vertex]
            return vertex

        acyclic = True
        for left, right in selected:
            left_root = root(left)
            right_root = root(right)
            if left_root == right_root:
                acyclic = False
                break
            parent[left_root] = right_root
        if acyclic and len({root(vertex) for vertex in graph.vertices}) == 1:
            count += 1
    return count


def test_flint_matrix_tree_matches_exhaustive_oracle_through_four_vertices() -> None:
    for order in range(5):
        vertices = tuple(str(index) for index in range(order))
        possible_edges = tuple(combinations(vertices, 2))
        for mask in range(1 << len(possible_edges)):
            graph = SimpleUndirectedGraph(
                vertices=vertices,
                edges=tuple(
                    edge
                    for index, edge in enumerate(possible_edges)
                    if mask & (1 << index)
                ),
            )
            result = _spanning_tree_count(build_simple_graph(graph), graph)
            assert result.spanning_tree_count == _brute_force_count(graph)
            assert result.connected == (
                order > 0 and (order == 1 or result.spanning_tree_count > 0)
            )


def test_complete_graph_count_agrees_with_cayley_formula() -> None:
    order = 12
    vertices = tuple(f"v{index:02d}" for index in range(order))
    graph = SimpleUndirectedGraph(
        vertices=vertices, edges=tuple(combinations(vertices, 2))
    )
    result = _spanning_tree_count(build_simple_graph(graph), graph)
    assert result.spanning_tree_count == order ** (order - 2)
