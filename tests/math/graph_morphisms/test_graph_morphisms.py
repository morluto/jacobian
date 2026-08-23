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


def _canonical_graph(
    vertices: list[str], edges: list[list[str]]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple(tuple(e) for e in edges),  # type: ignore[arg-type]
    )


class TestFixedLengthCycle:
    def _g(self, vertices, edges):
        return _canonical_graph(vertices, edges)

    def test_triangle_in_c4_with_chord(self):
        from jacobian.math.graphs.morphisms._models import FixedLengthCycleRequest
        from jacobian.math.graphs.morphisms._operations import (
            compute_fixed_length_cycle,
        )

        g = self._g(
            ["a", "b", "c", "d"],
            [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["c", "d"]],
        )
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

        g = self._g(
            ["a", "b", "c", "d"], [["a", "b"], ["a", "d"], ["b", "c"], ["c", "d"]]
        )
        result = compute_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        assert result.decision == "DOES_NOT_EXIST"
        assert result.cycle == ()

    def test_plain_c4_has_four_cycle(self):
        from jacobian.math.graphs.morphisms._models import FixedLengthCycleRequest
        from jacobian.math.graphs.morphisms._operations import (
            compute_fixed_length_cycle,
        )

        g = self._g(
            ["a", "b", "c", "d"], [["a", "b"], ["a", "d"], ["b", "c"], ["c", "d"]]
        )
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
        from jacobian.math.graphs.morphisms._operations import (
            compute_fixed_length_cycle,
        )
        from jacobian.math.graphs.operations import explicit_graph

        g = explicit_graph(
            vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c"), ("a", "c"))
        )
        # explicit_graph returns canonical SimpleUndirectedGraph; pass directly
        result = compute_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        assert result.decision == "EXISTS"

    def test_forged_negative_decision_is_rejected_by_replay(self):
        import pytest

        from jacobian.math.graphs.morphisms._models import FixedLengthCycleResult

        triangle = self._g(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
        with pytest.raises(ValueError, match="contradicts the retained"):
            FixedLengthCycleResult(
                graph=triangle, decision="DOES_NOT_EXIST", length=3, cycle=()
            )

    def test_oversized_length_is_rejected_before_exponentiating(self):
        import time

        import pytest

        from jacobian.math.graphs.morphisms._models import FixedLengthCycleResult

        # length has no schema upper bound on results, so the replay
        # validator must reject out-of-domain lengths before raising
        # d_max to that power.
        triangle = self._g(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
        start = time.monotonic()
        with pytest.raises(ValueError, match="vertex count"):
            FixedLengthCycleResult(
                graph=triangle,
                decision="DOES_NOT_EXIST",
                length=10_000_000_000,
                cycle=(),
            )
        assert time.monotonic() - start < 1.0

    def test_negative_decision_replays_inside_request_domain(self):
        from itertools import combinations

        import pytest

        from jacobian.math.graphs.morphisms._models import (
            MORPHISM_MAX_VERTICES,
            FixedLengthCycleRequest,
            FixedLengthCycleResult,
        )
        from jacobian.math.graphs.morphisms._operations import (
            compute_fixed_length_cycle,
        )

        # A path on 6 vertices has no triangle; the honest negative result
        # must validate through replay like any operation-produced value.
        path_edges = [[chr(ord("a") + i), chr(ord("a") + i + 1)] for i in range(5)]
        g = self._g(list("abcdef"), path_edges)
        result = compute_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        assert result.decision == "DOES_NOT_EXIST"
        revalidated = FixedLengthCycleResult(
            graph=g, decision=result.decision, length=result.length, cycle=result.cycle
        )
        assert revalidated.decision == "DOES_NOT_EXIST"

        # Outside the bounded request domain a negative decision is not
        # exact: reject an oversized retained graph with no witness check
        # to lean on.
        labels = [f"v{i}" for i in range(MORPHISM_MAX_VERTICES + 1)]
        big = self._g(labels, [list(e) for e in combinations(labels, 2)][:10])
        with pytest.raises(ValueError, match="request budget"):
            FixedLengthCycleResult(graph=big, decision="DOES_NOT_EXIST", length=3)

    def test_positive_witness_still_validates_beyond_search_domain(self):
        from jacobian.math.graphs.morphisms._models import FixedLengthCycleResult

        # An EXISTS conclusion is established by its witness alone; it stays
        # valid even when the source exceeds the searchable request domain.
        n = 24
        labels = [f"v{i}" for i in range(n)]
        cyc = [sorted((f"v{i}", f"v{(i + 1) % n}")) for i in range(n)]
        g = self._g(labels, cyc)
        result = FixedLengthCycleResult(
            graph=g,
            decision="EXISTS",
            length=n,
            cycle=tuple(labels),
        )
        assert result.decision == "EXISTS"

    def test_cycle_result_rejects_unsupported_budget_outcome(self):
        import pytest

        from jacobian.math.graphs.morphisms._models import FixedLengthCycleResult

        triangle = self._g(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
        with pytest.raises(ValueError, match="literal"):
            FixedLengthCycleResult(
                graph=triangle,
                decision="BUDGET_EXCEEDED",
                length=3,
                cycle=(),
            )


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
        host = self._g(
            ["a", "b", "c", "d"],
            [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["c", "d"]],
        )
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
        from jacobian.math.graphs.morphisms._operations import (
            compute_subgraph_pattern_find,
        )
        from jacobian.math.graphs.operations import explicit_graph

        pat = explicit_graph(vertices=("x", "y"), edges=(("x", "y"),))
        host = explicit_graph(vertices=("a", "b", "c"), edges=(("a", "b"), ("b", "c")))
        result = compute_subgraph_pattern_find(
            SubgraphPatternFindRequest(pattern=pat, host=host)
        )
        assert result.decision == "EXISTS"

    def test_request_admission_accounts_for_operation_and_replay_passes(self):
        import pytest

        from jacobian.math.graphs.morphisms._models import (
            MAX_CYCLE_SEARCH_PATHS,
            SubgraphPatternFindRequest,
        )

        # P(11, 8) = 6,652,800 assignments fits inside the advertised total
        # budget but not inside the per-pass share that also covers the
        # validator's replay of a negative decision, so it must be rejected.
        pat = self._g(
            [f"x{i}" for i in range(8)], [[f"x{i}", f"x{i + 1}"] for i in range(7)]
        )
        host_labels = [f"h{i:02d}" for i in range(11)]
        host = self._g(host_labels, [[f"h{i:02d}", f"h{i + 1:02d}"] for i in range(10)])
        assert MAX_CYCLE_SEARCH_PATHS // 2 < 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4
        with pytest.raises(ValueError, match="per-pass budget"):
            SubgraphPatternFindRequest(pattern=pat, host=host)

    def test_request_admission_reserves_output_headroom_for_source_echo(self):
        import pytest

        from jacobian.math.graphs.morphisms._models import FixedLengthCycleRequest

        # An edgeless 20-vertex graph with one multi-megabyte NFC label fits
        # the canonical input limit, but the result echoes the graph and adds
        # the envelope; admission must reject before any search runs.
        huge = "v" * (6 * 1024 * 1024)
        labels = [huge] + [f"w{i}" for i in range(19)]
        g = self._g(labels, [])
        with pytest.raises(ValueError, match="canonical output limit"):
            FixedLengthCycleRequest(graph=g, length=3)

    def test_forged_negative_decision_is_rejected_by_replay(self):
        import pytest

        from jacobian.math.graphs.morphisms._models import SubgraphPatternFindResult

        pat = self._g(["x", "y", "z"], [["x", "y"], ["x", "z"], ["y", "z"]])
        host = self._g(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
        with pytest.raises(ValueError, match="contradicts the retained"):
            SubgraphPatternFindResult(
                pattern=pat, host=host, decision="DOES_NOT_EXIST", vertex_map=()
            )

    def test_negative_decision_replays_inside_request_domain(self):
        import pytest

        from jacobian.math.graphs.morphisms._models import (
            MAX_CYCLE_SEARCH_PATHS,
            MORPHISM_MAX_VERTICES,
            SubgraphPatternFindRequest,
            SubgraphPatternFindResult,
        )
        from jacobian.math.graphs.morphisms._operations import (
            compute_subgraph_pattern_find,
        )

        # An honest negative: a triangle pattern cannot embed in a path.
        pat = self._g(["x", "y", "z"], [["x", "y"], ["x", "z"], ["y", "z"]])
        host = self._g(["a", "b", "c"], [["a", "b"], ["b", "c"]])
        result = compute_subgraph_pattern_find(
            SubgraphPatternFindRequest(pattern=pat, host=host),
        )
        assert result.decision == "DOES_NOT_EXIST"
        revalidated = SubgraphPatternFindResult(
            pattern=pat, host=host, decision=result.decision, vertex_map=()
        )
        assert revalidated.decision == "DOES_NOT_EXIST"

        # Outside the bounded request domain (pattern over the vertex cap)
        # a negative conclusion is not exact and must be rejected.
        big_pat_labels = [f"p{i:02d}" for i in range(MORPHISM_MAX_VERTICES + 1)]
        host_labels = [f"h{i:02d}" for i in range(MORPHISM_MAX_VERTICES + 1)]
        big_pat = self._g(
            big_pat_labels,
            [[f"p{i:02d}", f"p{i + 1:02d}"] for i in range(MORPHISM_MAX_VERTICES)],
        )
        big_host = self._g(
            host_labels,
            [[f"h{i:02d}", f"h{i + 1:02d}"] for i in range(MORPHISM_MAX_VERTICES)],
        )
        with pytest.raises(ValueError, match="request budget"):
            SubgraphPatternFindResult(
                pattern=big_pat, host=big_host, decision="DOES_NOT_EXIST", vertex_map=()
            )
        assert MAX_CYCLE_SEARCH_PATHS > 0


class TestSubgraphPatternFindLabelCost:
    def _g(self, vertices, edges):
        return _canonical_graph(vertices, edges)

    def test_long_shared_prefix_labels_decide_correctly(self):
        """Search work must be index work, not label-byte comparisons.

        Host labels share a long common prefix; lexicographic comparisons
        on such labels would multiply every assignment check by the label
        length and turn the admitted P(n,k) budget into an effectively
        unbounded run (review counterexample shape: matching pattern vs a
        host with fewer edges).
        """
        from jacobian.math.graphs.morphisms._models import (
            SubgraphPatternFindRequest,
            SubgraphPatternFindResult,
        )
        from jacobian.math.graphs.morphisms._operations import (
            compute_subgraph_pattern_find,
        )

        prefix = "n" * 4096
        pat = self._g(
            tuple(f"{prefix}p{i}" for i in range(10)),
            [[f"{prefix}p{i}", f"{prefix}p{5 + i}"] for i in range(5)],
        )
        # Four disjoint edges only: no 5-edge matching exists.
        host = self._g(
            tuple(f"{prefix}h{i}" for i in range(10)),
            [
                [f"{prefix}h0", f"{prefix}h1"],
                [f"{prefix}h2", f"{prefix}h3"],
                [f"{prefix}h4", f"{prefix}h5"],
                [f"{prefix}h6", f"{prefix}h7"],
            ],
        )
        result = compute_subgraph_pattern_find(
            SubgraphPatternFindRequest(pattern=pat, host=host),
        )
        assert result.decision == "DOES_NOT_EXIST"
        SubgraphPatternFindResult(
            pattern=pat, host=host, decision=result.decision, vertex_map=()
        )

        # The same host admits a 4-edge matching after dropping one edge.
        smaller_pattern = self._g(
            tuple(f"{prefix}q{i}" for i in range(8)),
            [[f"{prefix}q{i}", f"{prefix}q{4 + i}"] for i in range(4)],
        )
        found = compute_subgraph_pattern_find(
            SubgraphPatternFindRequest(pattern=smaller_pattern, host=host),
        )
        assert found.decision == "EXISTS"


class TestBacktrackingNodeBudget:
    def _complete(self, n):
        from jacobian.math.graphs.values import SimpleUndirectedGraph

        verts = tuple(f"{i:02d}" for i in range(1, n + 1))
        edges = tuple((a, b) for idx, a in enumerate(verts) for b in verts[idx + 1 :])
        return SimpleUndirectedGraph(vertices=verts, edges=edges)

    def test_internal_backtracking_nodes_are_charged_to_the_budget(self):
        """K10 into K10-minus-an-edge cannot return a free negative.

        A failed search visits 1,863,219 partial mappings and scans all ten
        host candidates at each one (~18.6M candidate checks, twice with the
        validation replay). The kernel charges those candidate checks to the
        per-pass budget and reports the typed non-conclusion instead of a
        negative decision established by a partially searched space.
        """
        from jacobian.math.graphs.morphisms._models import (
            SubgraphPatternFindRequest,
        )
        from jacobian.math.graphs.morphisms._operations import (
            compute_subgraph_pattern_find,
        )
        from jacobian.math.graphs.values import SimpleUndirectedGraph

        host_edges = tuple(
            edge for edge in self._complete(10).edges if edge != ("01", "02")
        )
        host = SimpleUndirectedGraph(
            vertices=self._complete(10).vertices, edges=host_edges
        )
        request = SubgraphPatternFindRequest(pattern=self._complete(10), host=host)
        result = compute_subgraph_pattern_find(request)
        assert result.decision == "BUDGET_EXCEEDED"
        assert result.vertex_map == ()
        # The typed non-conclusion round-trips without an exhaustive replay.
        type(result).model_validate(result.model_dump())

        import pytest

        with pytest.raises(ValueError, match="candidate-check budget"):
            type(result)(
                pattern=request.pattern,
                host=request.host,
                decision="DOES_NOT_EXIST",
                vertex_map=(),
            )
