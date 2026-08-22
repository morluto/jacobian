"""Tests for graph morphism operations."""

from jacobian.math.graphs.morphisms._models import (
    CoreCheckRequest,
    HomomorphismCheckRequest,
    HomomorphismFindRequest,
    RetractionCheckRequest,
    SimpleGraph,
)
from jacobian.math.graphs.morphisms._operations import (
    compute_core_check,
    compute_homomorphism_check,
    compute_homomorphism_find,
    compute_retraction_check,
)
from jacobian.math.graphs.morphisms._tools import TOOLS
from jacobian.math.graphs.values import SimpleUndirectedGraph


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "graph.core.check",
        "graph.cycle.fixed_length.decide",
        "graph.homomorphism.check",
        "graph.homomorphism.find",
        "graph.retraction.check",
        "graph.subgraph_pattern.find",
    }


def test_homomorphism_check_identity() -> None:
    request = HomomorphismCheckRequest(
        source_graph=SimpleGraph(vertex_count=2, edges=((0, 1),)),
        target_graph=SimpleGraph(vertex_count=2, edges=((0, 1),)),
        vertex_map=(0, 1),
    )
    result = compute_homomorphism_check(request)
    assert result.is_homomorphism is True


def test_homomorphism_check_non_homomorphism() -> None:
    request = HomomorphismCheckRequest(
        source_graph=SimpleGraph(vertex_count=2, edges=((0, 1),)),
        target_graph=SimpleGraph(vertex_count=2, edges=()),
        vertex_map=(0, 0),
    )
    result = compute_homomorphism_check(request)
    assert result.is_homomorphism is False


def test_homomorphism_find_k2_to_k2() -> None:
    request = HomomorphismFindRequest(
        source_graph=SimpleGraph(vertex_count=2, edges=((0, 1),)),
        target_graph=SimpleGraph(vertex_count=2, edges=((0, 1),)),
    )
    result = compute_homomorphism_find(request)
    assert result.found is True
    assert len(result.vertex_map) == 2


def test_homomorphism_find_no_homomorphism() -> None:
    request = HomomorphismFindRequest(
        source_graph=SimpleGraph(vertex_count=2, edges=((0, 1),)),
        target_graph=SimpleGraph(vertex_count=1, edges=()),
    )
    result = compute_homomorphism_find(request)
    assert result.found is False


def test_core_check_k2_is_core() -> None:
    request = CoreCheckRequest(graph=SimpleGraph(vertex_count=2, edges=((0, 1),)))
    result = compute_core_check(request)
    assert result.is_core is True


def test_core_check_independent_set_is_not_core() -> None:
    request = CoreCheckRequest(graph=SimpleGraph(vertex_count=3, edges=()))
    result = compute_core_check(request)
    assert result.is_core is False


def test_retraction_check_k3_to_edge() -> None:
    request = RetractionCheckRequest(
        graph=SimpleGraph(vertex_count=3, edges=((0, 1), (1, 2), (0, 2))),
        subgraph_vertices=(0, 1),
    )
    result = compute_retraction_check(request)
    assert result.is_retraction is False


def test_core_check_p3_is_not_core() -> None:
    """P3 (3-vertex path) retracts onto an edge, so it is not a core."""
    request = CoreCheckRequest(
        graph=SimpleGraph(vertex_count=3, edges=((0, 1), (1, 2)))
    )
    result = compute_core_check(request)
    assert result.is_core is False


def test_core_check_c4_is_not_core() -> None:
    """C4 (4-cycle) retracts onto an edge, so it is not a core."""
    request = CoreCheckRequest(
        graph=SimpleGraph(vertex_count=4, edges=((0, 1), (1, 2), (2, 3), (0, 3)))
    )
    result = compute_core_check(request)
    assert result.is_core is False


def test_core_check_k3_is_core() -> None:
    """K3 (complete graph on 3 vertices) is a core."""
    request = CoreCheckRequest(
        graph=SimpleGraph(vertex_count=3, edges=((0, 1), (1, 2), (0, 2)))
    )
    result = compute_core_check(request)
    assert result.is_core is True


def test_retraction_check_p3_to_edge() -> None:
    """P3 retracts onto the edge {0,1}: vertex 2 maps to 0."""
    request = RetractionCheckRequest(
        graph=SimpleGraph(vertex_count=3, edges=((0, 1), (1, 2))),
        subgraph_vertices=(0, 1),
    )
    result = compute_retraction_check(request)
    assert result.is_retraction is True


def test_core_check_c5_is_core() -> None:
    """C5 (5-cycle, odd) is a core: every endomorphism is an automorphism."""
    request = CoreCheckRequest(
        graph=SimpleGraph(
            vertex_count=5, edges=((0, 1), (1, 2), (2, 3), (3, 4), (4, 0))
        )
    )
    result = compute_core_check(request)
    assert result.is_core is True


def test_core_check_k4_is_core() -> None:
    """K4 (complete graph on 4 vertices) is a core."""
    request = CoreCheckRequest(
        graph=SimpleGraph(
            vertex_count=4,
            edges=((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        )
    )
    result = compute_core_check(request)
    assert result.is_core is True


def test_retraction_check_c4_to_edge() -> None:
    """C4 retracts onto any of its edges: vertex 2 maps to 0, vertex 3 maps to 1."""
    request = RetractionCheckRequest(
        graph=SimpleGraph(vertex_count=4, edges=((0, 1), (1, 2), (2, 3), (0, 3))),
        subgraph_vertices=(0, 1),
    )
    result = compute_retraction_check(request)
    assert result.is_retraction is True


def _canonical_graph(vertices: list[str], edges: list[list[str]]) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices), edges=tuple(tuple(e) for e in edges)  # type: ignore[arg-type]
    )


class TestFixedLengthCycle:
    def _g(self, vertices, edges):
        return _canonical_graph(vertices, edges)

    def test_triangle_in_c4_with_chord(self):
        from jacobian.math.graphs.morphisms._models import FixedLengthCycleRequest
        from jacobian.math.graphs.morphisms._operations import (
            compute_fixed_length_cycle,
        )

        g = self._g(["a", "b", "c", "d"], [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["c", "d"]])
        result = compute_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        assert result.decision == "EXISTS"
        assert len(result.cycle) == 3
        assert len(set(result.cycle)) == 3
        # verify the cycle closes via string edges
        adj = set()
        for u, v in g.edges:
            adj.add((u, v))
            adj.add((v, u))
        from itertools import pairwise

        cyc = [*list(result.cycle), result.cycle[0]]
        assert all((a, b) in adj for a, b in pairwise(cyc))
        # witness vertices must be from canonical graph
        assert all(v in g.vertices for v in result.cycle)

    def test_plain_c4_has_no_triangle(self):
        from jacobian.math.graphs.morphisms._models import FixedLengthCycleRequest
        from jacobian.math.graphs.morphisms._operations import (
            compute_fixed_length_cycle,
        )

        g = self._g(["a", "b", "c", "d"], [["a", "b"], ["a", "d"], ["b", "c"], ["c", "d"]])
        result = compute_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        assert result.decision == "DOES_NOT_EXIST"
        assert result.cycle == ()

    def test_plain_c4_has_four_cycle(self):
        from jacobian.math.graphs.morphisms._models import FixedLengthCycleRequest
        from jacobian.math.graphs.morphisms._operations import (
            compute_fixed_length_cycle,
        )

        g = self._g(["a", "b", "c", "d"], [["a", "b"], ["a", "d"], ["b", "c"], ["c", "d"]])
        result = compute_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=4))
        assert result.decision == "EXISTS"
        assert len(result.cycle) == 4

    def test_distinct_from_girth(self):
        # A graph with a 3-cycle and a 4-cycle: asking for length 4 still finds
        # the 4-cycle even though the girth is 3.
        from jacobian.math.graphs.morphisms._models import FixedLengthCycleRequest
        from jacobian.math.graphs.morphisms._operations import (
            compute_fixed_length_cycle,
        )

        g = self._g(
            ["a", "b", "c", "d"],
            [["a", "b"], ["a", "d"], ["b", "c"], ["a", "c"], ["c", "d"]],
        )
        r3 = compute_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        r4 = compute_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=4))
        assert r3.decision == "EXISTS"
        assert r4.decision == "EXISTS"

    def test_rejects_length_too_large(self):
        import pytest

        from jacobian.math.graphs.morphisms._models import FixedLengthCycleRequest

        g = self._g(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
        with pytest.raises(ValueError, match="vertex count"):
            FixedLengthCycleRequest(graph=g, length=4)

    def test_composes_with_canonical_graph(self):
        # Verify direct composition with graph API: explicit_graph output can be passed unchanged.
        from jacobian.math.graphs.morphisms._models import FixedLengthCycleRequest
        from jacobian.math.graphs.morphisms._operations import compute_fixed_length_cycle
        from jacobian.math.graphs.operations import explicit_graph

        g = explicit_graph(
            vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c"), ("a", "c"))
        )
        # explicit_graph returns canonical SimpleUndirectedGraph; pass directly
        result = compute_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        assert result.decision == "EXISTS"


class TestSubgraphPatternFind:
    def _g(self, vertices, edges):
        return _canonical_graph(vertices, edges)

    def test_triangle_embeds_in_c4_with_chord(self):
        from jacobian.math.graphs.morphisms._models import (
            SubgraphPatternFindRequest,
        )
        from jacobian.math.graphs.morphisms._operations import (
            compute_subgraph_pattern_find,
        )

        pat = self._g(["x", "y", "z"], [["x", "y"], ["x", "z"], ["y", "z"]])
        host = self._g(["a", "b", "c", "d"], [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["c", "d"]])
        result = compute_subgraph_pattern_find(
            SubgraphPatternFindRequest(pattern=pat, host=host),
        )
        assert result.decision == "EXISTS"
        m = result.vertex_map
        assert len(set(m)) == 3  # injective
        # m is ordered by pattern vertex order: pattern.vertices = (x,y,z)
        pat_vertices = pat.vertices
        host_edges = set()
        for u, v in host.edges:
            host_edges.add((u, v))
            host_edges.add((v, u))
        # Build map from pattern label to host label
        mapping = {pat_vertices[i]: m[i] for i in range(len(pat_vertices))}
        for u, v in pat.edges:
            assert (mapping[u], mapping[v]) in host_edges

    def test_p3_not_in_matching(self):
        from jacobian.math.graphs.morphisms._models import (
            SubgraphPatternFindRequest,
        )
        from jacobian.math.graphs.morphisms._operations import (
            compute_subgraph_pattern_find,
        )

        pat = self._g(["x", "y", "z"], [["x", "y"], ["y", "z"]])
        host = self._g(["a", "b", "c", "d"], [["a", "b"], ["c", "d"]])
        result = compute_subgraph_pattern_find(
            SubgraphPatternFindRequest(pattern=pat, host=host),
        )
        assert result.decision == "DOES_NOT_EXIST"
        assert result.vertex_map == ()

    def test_non_induced_allows_chords(self):
        # A triangle pattern embeds in a K4 host (which has chords) — ordinary,
        # non-induced containment.
        from jacobian.math.graphs.morphisms._models import (
            SubgraphPatternFindRequest,
        )
        from jacobian.math.graphs.morphisms._operations import (
            compute_subgraph_pattern_find,
        )

        pat = self._g(["x", "y", "z"], [["x", "y"], ["x", "z"], ["y", "z"]])
        host = self._g(
            ["a", "b", "c", "d"],
            [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["b", "d"], ["c", "d"]],
        )
        result = compute_subgraph_pattern_find(
            SubgraphPatternFindRequest(pattern=pat, host=host),
        )
        assert result.decision == "EXISTS"

    def test_rejects_pattern_larger_than_host(self):
        import pytest

        from jacobian.math.graphs.morphisms._models import (
            SubgraphPatternFindRequest,
        )

        pat = self._g(["x", "y", "z"], [["x", "y"], ["y", "z"]])
        host = self._g(["a", "b"], [["a", "b"]])
        with pytest.raises(ValueError, match="more vertices"):
            SubgraphPatternFindRequest(pattern=pat, host=host)

    def test_composes_with_canonical_graph(self):
        from jacobian.math.graphs.morphisms._models import SubgraphPatternFindRequest
        from jacobian.math.graphs.morphisms._operations import compute_subgraph_pattern_find
        from jacobian.math.graphs.operations import explicit_graph

        pat = explicit_graph(vertices=("x", "y"), edges=(("x", "y"),))
        host = explicit_graph(vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c")))
        result = compute_subgraph_pattern_find(SubgraphPatternFindRequest(pattern=pat, host=host))
        assert result.decision == "EXISTS"
