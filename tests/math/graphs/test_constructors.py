"""Contract tests for graph constructor operations."""

from __future__ import annotations

import pytest
import itertools

from jacobian.math.graphs.constructors._models import (
    HypercubeGraphRequest,
    KellerGraphRequest,
    TriangleProfileRequest,
)
from jacobian.math.graphs.constructors._operations import (
    construct_hypercube_graph,
    construct_keller_graph,
    compute_triangle_profile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


class TestHypercubeGraph:
    def test_q0(self) -> None:
        """Q_0: one vertex, no edges."""
        result = construct_hypercube_graph(HypercubeGraphRequest(dimension=0))
        assert result.graph.vertex_count == 1
        assert len(result.graph.edges) == 0

    def test_q1(self) -> None:
        """Q_1: two vertices, one edge."""
        result = construct_hypercube_graph(HypercubeGraphRequest(dimension=1))
        assert result.graph.vertex_count == 2
        assert len(result.graph.edges) == 1

    def test_q3(self) -> None:
        """Q_3: 8 vertices, 12 edges."""
        result = construct_hypercube_graph(HypercubeGraphRequest(dimension=3))
        assert result.graph.vertex_count == 8
        assert len(result.graph.edges) == 12

    def test_edge_count_formula(self) -> None:
        """Q_d has d * 2^(d-1) edges."""
        for d in range(0, 7):
            result = construct_hypercube_graph(HypercubeGraphRequest(dimension=d))
            expected_edges = d * (2 ** (d - 1)) if d > 0 else 0
            assert len(result.graph.edges) == expected_edges

    def test_vertex_count_formula(self) -> None:
        """Q_d has 2^d vertices."""
        for d in range(0, 8):
            result = construct_hypercube_graph(HypercubeGraphRequest(dimension=d))
            assert result.graph.vertex_count == 2 ** d

    def test_adjacency_correctness(self) -> None:
        """Two vertices in Q_d are adjacent iff they differ in exactly one bit."""
        d = 4
        result = construct_hypercube_graph(HypercubeGraphRequest(dimension=d))
        edge_set = set(result.graph.edges)
        n = 2 ** d
        for i in range(n):
            for j in range(i + 1, n):
                diff = i ^ j
                expected_adjacent = diff != 0 and (diff & (diff - 1)) == 0
                actual_adjacent = (i, j) in edge_set
                assert actual_adjacent == expected_adjacent

    def test_edges_are_ordered(self) -> None:
        """All edges should have left < right."""
        result = construct_hypercube_graph(HypercubeGraphRequest(dimension=3))
        for u, v in result.graph.edges:
            assert u < v

    def test_no_duplicate_edges(self) -> None:
        result = construct_hypercube_graph(HypercubeGraphRequest(dimension=4))
        edges = result.graph.edges
        assert len(edges) == len(set(edges))


class TestKellerGraph:
    def test_k0(self) -> None:
        """K_0: one vertex, no edges."""
        result = construct_keller_graph(KellerGraphRequest(dimension=0))
        assert result.graph.vertex_count == 1
        assert len(result.graph.edges) == 0

    def test_k1(self) -> None:
        """K_1: 4 vertices, no edges (distinct words differ in only 1 coordinate)."""
        result = construct_keller_graph(KellerGraphRequest(dimension=1))
        assert result.graph.vertex_count == 4
        assert len(result.graph.edges) == 0

    def test_k2(self) -> None:
        """K_2: 16 vertices, 40 edges."""
        result = construct_keller_graph(KellerGraphRequest(dimension=2))
        assert result.graph.vertex_count == 16
        assert len(result.graph.edges) == 40

    def test_vertex_count_formula(self) -> None:
        """K_d has 4^d vertices."""
        for d in range(0, 4):
            result = construct_keller_graph(KellerGraphRequest(dimension=d))
            assert result.graph.vertex_count == 4 ** d

    def test_adjacency_correctness(self) -> None:
        """Two words are adjacent iff they differ by 2 (mod 4) in some coordinate
        AND have Hamming distance >= 2."""
        d = 3
        result = construct_keller_graph(KellerGraphRequest(dimension=d))
        edge_set = set(result.graph.edges)
        n = 4 ** d
        for i in range(n):
            for j in range(i + 1, n):
                wi = _to_word(i, d)
                wj = _to_word(j, d)
                expected = _keller_adjacent(wi, wj)
                actual = (i, j) in edge_set
                assert actual == expected

    def test_no_loops_or_duplicates(self) -> None:
        result = construct_keller_graph(KellerGraphRequest(dimension=3))
        for u, v in result.graph.edges:
            assert u < v
        edges = result.graph.edges
        assert len(edges) == len(set(edges))


class TestTriangleProfile:
    def test_k4(self) -> None:
        """K_4 has 4 triangles."""
        graph = SimpleUndirectedGraph(
            vertices=("a", "b", "c", "d"),
            edges=(("a", "b"), ("a", "c"), ("a", "d"),
                   ("b", "c"), ("b", "d"), ("c", "d")),
        )
        result = compute_triangle_profile(TriangleProfileRequest(graph=graph))
        assert result.triangle_count == 4
        assert len(result.triangles) == 4

    def test_empty_graph(self) -> None:
        """A graph with no edges has no triangles."""
        graph = SimpleUndirectedGraph(
            vertices=("a", "b", "c"),
            edges=(),
        )
        result = compute_triangle_profile(TriangleProfileRequest(graph=graph))
        assert result.triangle_count == 0

    def test_single_triangle(self) -> None:
        graph = SimpleUndirectedGraph(
            vertices=("a", "b", "c"),
            edges=(("a", "b"), ("b", "c"), ("a", "c")),
        )
        result = compute_triangle_profile(TriangleProfileRequest(graph=graph))
        assert result.triangle_count == 1
        assert result.triangles[0].vertices == ("a", "b", "c")

    def test_path_graph(self) -> None:
        """A path graph has no triangles."""
        graph = SimpleUndirectedGraph(
            vertices=("a", "b", "c", "d"),
            edges=(("a", "b"), ("b", "c"), ("c", "d")),
        )
        result = compute_triangle_profile(TriangleProfileRequest(graph=graph))
        assert result.triangle_count == 0

    def test_complete_graph_k5(self) -> None:
        """K_5 has C(5,3) = 10 triangles."""
        vertices = tuple("abcde")
        edges = tuple(
            (vertices[i], vertices[j])
            for i in range(5) for j in range(i + 1, 5)
        )
        graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)
        result = compute_triangle_profile(TriangleProfileRequest(graph=graph))
        assert result.triangle_count == 10

    def test_triangle_replays_source_edges(self) -> None:
        """Every triangle's three edges exist in the source graph."""
        graph = SimpleUndirectedGraph(
            vertices=("a", "b", "c", "d"),
            edges=(("a", "b"), ("a", "c"), ("a", "d"),
                   ("b", "c"), ("b", "d"), ("c", "d")),
        )
        edge_set = {frozenset(e) for e in graph.edges}
        result = compute_triangle_profile(TriangleProfileRequest(graph=graph))
        for tri in result.triangles:
            v = tri.vertices
            assert frozenset((v[0], v[1])) in edge_set
            assert frozenset((v[0], v[2])) in edge_set
            assert frozenset((v[1], v[2])) in edge_set

    def test_no_shared_edges_become_shared_hypergraph(self) -> None:
        """Triangles sharing an edge should both appear (edge is shared)."""
        graph = SimpleUndirectedGraph(
            vertices=("a", "b", "c", "d"),
            edges=(("a", "b"), ("a", "c"), ("a", "d"),
                   ("b", "c"), ("c", "d")),
        )
        result = compute_triangle_profile(TriangleProfileRequest(graph=graph))
        # Triangles: (a,b,c) and (a,c,d)
        assert result.triangle_count == 2


def _to_word(n: int, d: int) -> tuple[int, ...]:
    word = []
    for _ in range(d):
        word.append(n % 4)
        n //= 4
    return tuple(reversed(word))


def _keller_adjacent(wi: tuple[int, ...], wj: tuple[int, ...]) -> bool:
    has_diff_2_mod_4 = False
    hamming = 0
    for a, b in zip(wi, wj, strict=True):
        if a != b:
            hamming += 1
            if abs(a - b) == 2:
                has_diff_2_mod_4 = True
    return has_diff_2_mod_4 and hamming >= 2
