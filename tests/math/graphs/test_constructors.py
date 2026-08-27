"""Contract tests for graph constructor operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.constructors import (
    compute_triangle_profile,
    construct_hypercube_graph,
    construct_keller_graph,
)
from jacobian.math.graphs.constructors._models import (
    TriangleProfileRequest,
    TriangleProfileResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


class TestHypercubeGraph:
    def test_q0(self) -> None:
        """Q_0: one vertex, no edges."""
        result = construct_hypercube_graph(0)
        assert result.graph.vertex_count == 1
        assert len(result.graph.edges) == 0

    def test_q1(self) -> None:
        """Q_1: two vertices, one edge."""
        result = construct_hypercube_graph(1)
        assert result.graph.vertex_count == 2
        assert len(result.graph.edges) == 1

    def test_q3(self) -> None:
        """Q_3: 8 vertices, 12 edges."""
        result = construct_hypercube_graph(3)
        assert result.graph.vertex_count == 8
        assert len(result.graph.edges) == 12

    def test_edge_count_formula(self) -> None:
        """Q_d has d * 2^(d-1) edges."""
        for d in range(7):
            result = construct_hypercube_graph(d)
            expected_edges = d * (2 ** (d - 1)) if d > 0 else 0
            assert len(result.graph.edges) == expected_edges

    def test_vertex_count_formula(self) -> None:
        """Q_d has 2^d vertices."""
        for d in range(8):
            result = construct_hypercube_graph(d)
            assert result.graph.vertex_count == 2**d

    def test_adjacency_correctness(self) -> None:
        """Two vertices in Q_d are adjacent iff they differ in exactly one bit."""
        d = 4
        result = construct_hypercube_graph(d)
        edge_set = set(result.graph.edges)
        n = 2**d
        for i in range(n):
            for j in range(i + 1, n):
                diff = i ^ j
                expected_adjacent = diff != 0 and (diff & (diff - 1)) == 0
                actual_adjacent = (i, j) in edge_set
                assert actual_adjacent == expected_adjacent

    def test_edges_are_ordered(self) -> None:
        """All edges should have left < right."""
        result = construct_hypercube_graph(3)
        for u, v in result.graph.edges:
            assert u < v

    def test_no_duplicate_edges(self) -> None:
        result = construct_hypercube_graph(4)
        edges = result.graph.edges
        assert len(edges) == len(set(edges))


class TestKellerGraph:
    def test_k0(self) -> None:
        """K_0: one vertex, no edges."""
        result = construct_keller_graph(0)
        assert result.graph.vertex_count == 1
        assert len(result.graph.edges) == 0

    def test_k1(self) -> None:
        """K_1: 4 vertices, no edges (distinct words differ in only 1 coordinate)."""
        result = construct_keller_graph(1)
        assert result.graph.vertex_count == 4
        assert len(result.graph.edges) == 0

    def test_k2(self) -> None:
        """K_2: 16 vertices, 40 edges."""
        result = construct_keller_graph(2)
        assert result.graph.vertex_count == 16
        assert len(result.graph.edges) == 40

    def test_vertex_count_formula(self) -> None:
        """K_d has 4^d vertices."""
        for d in range(4):
            result = construct_keller_graph(d)
            assert result.graph.vertex_count == 4**d

    def test_adjacency_correctness(self) -> None:
        """Two words are adjacent iff they differ by 2 (mod 4) in some coordinate
        AND have Hamming distance >= 2."""
        d = 3
        result = construct_keller_graph(d)
        edge_set = set(result.graph.edges)
        n = 4**d
        for i in range(n):
            for j in range(i + 1, n):
                wi = _to_word(i, d)
                wj = _to_word(j, d)
                expected = _keller_adjacent(wi, wj)
                actual = (i, j) in edge_set
                assert actual == expected

    def test_no_loops_or_duplicates(self) -> None:
        result = construct_keller_graph(3)
        for u, v in result.graph.edges:
            assert u < v
        edges = result.graph.edges
        assert len(edges) == len(set(edges))


class TestTriangleProfile:
    def test_k4(self) -> None:
        """K_4 has 4 triangles."""
        graph = SimpleUndirectedGraph(
            vertices=("a", "b", "c", "d"),
            edges=(
                ("a", "b"),
                ("a", "c"),
                ("a", "d"),
                ("b", "c"),
                ("b", "d"),
                ("c", "d"),
            ),
        )
        result = compute_triangle_profile(graph)
        assert result.triangle_count == 4
        assert len(result.triangles) == 4
        assert isinstance(result.triangles, tuple)

    def test_dense_graph_is_rejected_by_output_budget(self) -> None:
        vertices = tuple(f"{index:03d}" for index in range(256))
        graph = SimpleUndirectedGraph(
            vertices=vertices,
            edges=tuple(
                (vertices[left], vertices[right])
                for left in range(len(vertices))
                for right in range(left + 1, len(vertices))
            ),
        )

        with pytest.raises(OperationDomainValidationError) as caught:
            compute_triangle_profile(graph)
        error = caught.value.errors()[0]
        assert error["loc"] == ("graph",)
        assert error["type"] == "graph.triangle_profile.output_budget"

    def test_large_edgeless_graph_is_admitted_by_actual_work(self) -> None:
        """An edgeless graph has no candidate triangle rows to materialize."""
        vertices = tuple(f"v{index:03d}" for index in range(256))
        graph = SimpleUndirectedGraph(vertices=vertices, edges=())

        result = compute_triangle_profile(graph)

        assert result.source == graph
        assert result.triangles == ()
        assert result.triangle_count == 0

    def test_large_sparse_graph_is_admitted_by_actual_triangle_output(self) -> None:
        """A 256-vertex graph with one triangle stays inside its real envelope."""
        vertices = tuple(f"v{index:03d}" for index in range(256))
        graph = SimpleUndirectedGraph(
            vertices=vertices,
            edges=(
                (vertices[0], vertices[1]),
                (vertices[0], vertices[2]),
                (vertices[1], vertices[2]),
            ),
        )

        result = compute_triangle_profile(graph)

        assert result.triangle_count == 1
        assert result.triangles[0].vertices == vertices[:3]

    def test_long_labels_are_charged_in_dense_output_bound(self) -> None:
        """A K_100 with maximum-length labels cannot fit its retained rows."""
        vertices = tuple(f"{index:03d}" + "x" * 61 for index in range(100))
        graph = SimpleUndirectedGraph(
            vertices=vertices,
            edges=tuple(
                (vertices[left], vertices[right])
                for left in range(len(vertices))
                for right in range(left + 1, len(vertices))
            ),
        )

        with pytest.raises(ValueError, match=r"labelled rows.*canonical output budget"):
            compute_triangle_profile(graph)

    def test_request_construction_defers_result_admission(self) -> None:
        """Request parsing does not enumerate K_100's triangle rows."""
        vertices = tuple(f"{index:03d}" + "x" * 61 for index in range(100))
        graph = SimpleUndirectedGraph(
            vertices=vertices,
            edges=tuple(
                (vertices[left], vertices[right])
                for left in range(len(vertices))
                for right in range(left + 1, len(vertices))
            ),
        )

        request = TriangleProfileRequest(graph=graph)

        assert request.graph == graph
        with pytest.raises(ValueError, match=r"labelled rows.*canonical output budget"):
            compute_triangle_profile(request.graph)

    def test_short_labels_can_admit_the_same_dense_graph(self) -> None:
        """The label-aware bound admits K_100 when its actual rows fit."""
        vertices = tuple(f"v{index:02d}" for index in range(100))
        graph = SimpleUndirectedGraph(
            vertices=vertices,
            edges=tuple(
                (vertices[left], vertices[right])
                for left in range(len(vertices))
                for right in range(left + 1, len(vertices))
            ),
        )

        result = compute_triangle_profile(graph)

        assert result.triangle_count == 161_700
        assert (
            len(encode_strict_json(result.model_dump(mode="json")))
            <= CanonicalLimits().max_output_bytes
        )

    def test_native_wrapper_rejects_wire_request_values(self) -> None:
        """Native callers pass a graph value, not a request envelope."""
        graph = SimpleUndirectedGraph(vertices=("a", "b", "c"), edges=())

        with pytest.raises(TypeError, match="SimpleUndirectedGraph"):
            compute_triangle_profile({"graph": graph})  # type: ignore[arg-type]

    def test_triangle_count_rejects_negative_values(self) -> None:
        graph = SimpleUndirectedGraph(vertices=("a", "b", "c"), edges=())

        with pytest.raises(ValidationError) as caught:
            TriangleProfileResult(source=graph, triangles=(), triangle_count=-1)

        assert caught.value.errors()[0]["type"] == "greater_than_equal"

    def test_triangle_count_must_bind_to_returned_rows(self) -> None:
        graph = SimpleUndirectedGraph(vertices=("a", "b", "c"), edges=())

        with pytest.raises(ValidationError) as caught:
            TriangleProfileResult(source=graph, triangles=(), triangle_count=7)

        assert caught.value.errors()[0]["type"] == (
            "graph.triangle_profile.count_mismatch"
        )

    def test_empty_graph(self) -> None:
        """A graph with no edges has no triangles."""
        graph = SimpleUndirectedGraph(
            vertices=("a", "b", "c"),
            edges=(),
        )
        result = compute_triangle_profile(graph)
        assert result.triangle_count == 0

    def test_single_triangle(self) -> None:
        graph = SimpleUndirectedGraph(
            vertices=("a", "b", "c"),
            edges=(("a", "b"), ("b", "c"), ("a", "c")),
        )
        result = compute_triangle_profile(graph)
        assert result.triangle_count == 1
        assert result.triangles[0].vertices == ("a", "b", "c")

    def test_path_graph(self) -> None:
        """A path graph has no triangles."""
        graph = SimpleUndirectedGraph(
            vertices=("a", "b", "c", "d"),
            edges=(("a", "b"), ("b", "c"), ("c", "d")),
        )
        result = compute_triangle_profile(graph)
        assert result.triangle_count == 0

    def test_complete_graph_k5(self) -> None:
        """K_5 has C(5,3) = 10 triangles."""
        vertices = tuple("abcde")
        edges = tuple(
            (vertices[i], vertices[j]) for i in range(5) for j in range(i + 1, 5)
        )
        graph = SimpleUndirectedGraph(vertices=vertices, edges=edges)
        result = compute_triangle_profile(graph)
        assert result.triangle_count == 10

    def test_triangle_replays_source_edges(self) -> None:
        """Every triangle's three edges exist in the source graph."""
        graph = SimpleUndirectedGraph(
            vertices=("a", "b", "c", "d"),
            edges=(
                ("a", "b"),
                ("a", "c"),
                ("a", "d"),
                ("b", "c"),
                ("b", "d"),
                ("c", "d"),
            ),
        )
        edge_set = {frozenset(e) for e in graph.edges}
        result = compute_triangle_profile(graph)
        for tri in result.triangles:
            v = tri.vertices
            assert frozenset((v[0], v[1])) in edge_set
            assert frozenset((v[0], v[2])) in edge_set
            assert frozenset((v[1], v[2])) in edge_set

    def test_no_shared_edges_become_shared_hypergraph(self) -> None:
        """Triangles sharing an edge should both appear (edge is shared)."""
        graph = SimpleUndirectedGraph(
            vertices=("a", "b", "c", "d"),
            edges=(("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("c", "d")),
        )
        result = compute_triangle_profile(graph)
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
