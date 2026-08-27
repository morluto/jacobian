"""Tests for graph morphism operations."""

import pytest
from pydantic import ValidationError

from jacobian.canonical import CanonicalLimits
from jacobian.math.graphs.morphisms import _models as morphism_models
from jacobian.math.graphs.morphisms._models import (
    GraphHomomorphism,
    GraphHomomorphismObstruction,
    GraphVertexMap,
    GraphVertexMapRow,
    HomomorphismCheckRequest,
    HomomorphismCheckResult,
)
from jacobian.math.graphs.morphisms._operations import (
    compute_homomorphism_check,
    verify_homomorphism_check_result,
)
from jacobian.math.graphs.morphisms._tools import TOOLS
from jacobian.math.graphs.values import SimpleUndirectedGraph


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "graph.cycle.fixed_length.decide",
        "graph.homomorphism.check",
        "graph.subgraph_pattern.find",
    }


def _vertex_map(
    source_vertices: tuple[str, ...],
    source_edges: tuple[tuple[str, str], ...],
    target_vertices: tuple[str, ...],
    target_edges: tuple[tuple[str, str], ...],
    rows: tuple[tuple[str, str], ...],
) -> GraphVertexMap:
    return GraphVertexMap(
        source_graph=SimpleUndirectedGraph(
            vertices=source_vertices,
            edges=source_edges,
        ),
        target_graph=SimpleUndirectedGraph(
            vertices=target_vertices,
            edges=target_edges,
        ),
        rows=tuple(
            GraphVertexMapRow(source_vertex=source, target_vertex=target)
            for source, target in rows
        ),
    )


def test_homomorphism_check_returns_source_bound_checked_map() -> None:
    """Rows are canonical by label even when a graph's display order is not."""

    vertex_map = _vertex_map(
        ("b", "a"),
        (("a", "b"),),
        ("y", "x"),
        (("x", "y"),),
        (("a", "x"), ("b", "y")),
    )
    result = compute_homomorphism_check(HomomorphismCheckRequest(vertex_map=vertex_map))
    assert result.status == "HOMOMORPHISM"
    assert result.obstruction is None
    assert result.homomorphism == GraphHomomorphism(vertex_map=vertex_map)


def test_homomorphism_check_returns_first_edge_image_nonedge() -> None:
    vertex_map = _vertex_map(
        ("a", "b", "c"),
        (("a", "b"), ("a", "c")),
        ("x", "y"),
        (("x", "y"),),
        (("a", "x"), ("b", "x"), ("c", "y")),
    )
    result = compute_homomorphism_check(HomomorphismCheckRequest(vertex_map=vertex_map))
    assert result.status == "EDGE_IMAGE_NOT_EDGE"
    assert result.homomorphism is None
    assert result.obstruction == GraphHomomorphismObstruction(
        vertex_map=vertex_map,
        source_edge=("a", "b"),
        image_vertices=("x", "x"),
    )


def test_homomorphism_check_accepts_edgeless_noninjective_map() -> None:
    vertex_map = _vertex_map(
        ("a", "b"),
        (),
        ("x",),
        (),
        (("a", "x"), ("b", "x")),
    )
    result = compute_homomorphism_check(HomomorphismCheckRequest(vertex_map=vertex_map))
    assert result.status == "HOMOMORPHISM"
    assert result.homomorphism is not None


def test_vertex_map_rejects_incomplete_out_of_order_and_foreign_rows() -> None:
    source = SimpleUndirectedGraph(vertices=("a", "b"), edges=(("a", "b"),))
    target = SimpleUndirectedGraph(vertices=("x", "y"), edges=(("x", "y"),))

    with pytest.raises(ValidationError):
        GraphVertexMap(
            source_graph=source,
            target_graph=target,
            rows=(GraphVertexMapRow(source_vertex="a", target_vertex="x"),),
        )
    with pytest.raises(ValidationError):
        GraphVertexMap(
            source_graph=source,
            target_graph=target,
            rows=(
                GraphVertexMapRow(source_vertex="b", target_vertex="y"),
                GraphVertexMapRow(source_vertex="a", target_vertex="x"),
            ),
        )
    with pytest.raises(ValidationError):
        GraphVertexMap(
            source_graph=source,
            target_graph=target,
            rows=(
                GraphVertexMapRow(source_vertex="a", target_vertex="x"),
                GraphVertexMapRow(source_vertex="b", target_vertex="z"),
            ),
        )


def test_homomorphism_result_is_structural_and_verifier_rejects_forged_claim() -> None:
    forged = HomomorphismCheckResult(
        status="EDGE_IMAGE_NOT_EDGE",
        obstruction=GraphHomomorphismObstruction(
            vertex_map=_vertex_map(
                ("a", "b"),
                (("a", "b"),),
                ("x", "y"),
                (("x", "y"),),
                (("a", "x"), ("b", "y")),
            ),
            source_edge=("a", "b"),
            image_vertices=("x", "y"),
        ),
    )
    assert not verify_homomorphism_check_result(forged)

    with pytest.raises(ValidationError):
        HomomorphismCheckResult(
            status="EDGE_IMAGE_NOT_EDGE",
        )


def test_homomorphism_check_orders_edge_obstructions_canonically() -> None:
    def first_obstruction(
        source_edges: tuple[tuple[str, str], ...],
    ) -> GraphHomomorphismObstruction:
        vertex_map = _vertex_map(
            ("a", "b", "c"),
            source_edges,
            ("x", "y"),
            (("x", "y"),),
            (("a", "x"), ("b", "x"), ("c", "x")),
        )
        result = compute_homomorphism_check(
            HomomorphismCheckRequest(vertex_map=vertex_map)
        )
        assert result.obstruction is not None
        return result.obstruction

    assert first_obstruction((("a", "b"), ("a", "c"))).source_edge == (
        "a",
        "b",
    )
    assert first_obstruction((("a", "c"), ("a", "b"))).source_edge == (
        "a",
        "b",
    )


def test_homomorphism_check_preflights_retained_result_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_limit = CanonicalLimits(max_output_bytes=400)
    monkeypatch.setattr(morphism_models, "CanonicalLimits", lambda: small_limit)

    with pytest.raises(ValidationError):
        _vertex_map(
            ("a" * 100,),
            (),
            ("b" * 100,),
            (),
            (("a" * 100, "b" * 100),),
        )


def _canonical_graph(
    vertices: list[str] | tuple[str, ...],
    edges: list[list[str]] | tuple[tuple[str, str], ...],
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((edge[0], edge[1]) for edge in edges),
    )


class TestFixedLengthCycle:
    def _g(
        self,
        vertices: list[str] | tuple[str, ...],
        edges: list[list[str]] | tuple[tuple[str, str], ...],
    ) -> SimpleUndirectedGraph:
        return _canonical_graph(vertices, edges)

    def test_triangle_in_c4_with_chord(self) -> None:
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

    def test_plain_c4_has_no_triangle(self) -> None:
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

    def test_plain_c4_has_four_cycle(self) -> None:
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

    def test_distinct_from_girth(self) -> None:
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

    def test_rejects_length_too_large(self) -> None:
        import pytest

        from jacobian.math.graphs.morphisms._models import FixedLengthCycleRequest

        g = self._g(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
        with pytest.raises(ValueError):
            FixedLengthCycleRequest(graph=g, length=4)

    def test_composes_with_canonical_graph(self) -> None:
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

    def test_forged_negative_decision_is_rejected_by_explicit_verifier(self) -> None:
        from jacobian.math.graphs.morphisms._models import FixedLengthCycleResult
        from jacobian.math.graphs.morphisms._operations import (
            verify_fixed_length_cycle_result,
        )

        triangle = self._g(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
        forged = FixedLengthCycleResult(
            graph=triangle, decision="DOES_NOT_EXIST", length=3, cycle=()
        )
        assert not verify_fixed_length_cycle_result(forged)

    def test_oversized_length_is_rejected_before_exponentiating(self) -> None:
        import time

        import pytest

        from jacobian.math.graphs.morphisms._models import FixedLengthCycleResult

        # Result parsing keeps the negative request envelope bounded before
        # raising d_max to an untrusted exponent.
        triangle = self._g(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
        start = time.monotonic()
        with pytest.raises(ValueError):
            FixedLengthCycleResult(
                graph=triangle,
                decision="DOES_NOT_EXIST",
                length=10_000_000_000,
                cycle=(),
            )
        assert time.monotonic() - start < 1.0

    def test_negative_decision_is_structural_inside_request_domain(self) -> None:
        from itertools import combinations

        import pytest

        from jacobian.math.graphs.morphisms._models import (
            MORPHISM_MAX_VERTICES,
            FixedLengthCycleRequest,
            FixedLengthCycleResult,
        )
        from jacobian.math.graphs.morphisms._operations import (
            compute_fixed_length_cycle,
            verify_fixed_length_cycle_result,
        )

        # A path on 6 vertices has no triangle; the honest negative result
        # round-trips structurally and the owner verifier checks its claim.
        path_edges = [[chr(ord("a") + i), chr(ord("a") + i + 1)] for i in range(5)]
        g = self._g(list("abcdef"), path_edges)
        result = compute_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        assert result.decision == "DOES_NOT_EXIST"
        revalidated = FixedLengthCycleResult(
            graph=g, decision=result.decision, length=result.length, cycle=result.cycle
        )
        assert revalidated.decision == "DOES_NOT_EXIST"
        assert verify_fixed_length_cycle_result(revalidated)

        # Outside the bounded request domain a negative decision is not
        # exact: reject an oversized retained graph with no witness check
        # to lean on.
        labels = [f"v{i}" for i in range(MORPHISM_MAX_VERTICES + 1)]
        big = self._g(labels, [list(e) for e in combinations(labels, 2)][:10])
        with pytest.raises(ValueError):
            FixedLengthCycleResult(graph=big, decision="DOES_NOT_EXIST", length=3)

    def test_positive_witness_still_validates_beyond_search_domain(self) -> None:
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

    def test_cycle_result_rejects_unsupported_budget_outcome(self) -> None:
        import pytest

        from jacobian.math.graphs.morphisms._models import FixedLengthCycleResult

        triangle = self._g(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
        with pytest.raises(ValueError):
            FixedLengthCycleResult.model_validate(
                {
                    "graph": triangle.model_dump(mode="json"),
                    "decision": "BUDGET_EXCEEDED",
                    "length": 3,
                    "cycle": [],
                }
            )


class TestSubgraphPatternFind:
    def _g(
        self,
        vertices: list[str] | tuple[str, ...],
        edges: list[list[str]] | tuple[tuple[str, str], ...],
    ) -> SimpleUndirectedGraph:
        return _canonical_graph(vertices, edges)

    def test_triangle_embeds_in_c4_with_chord(self) -> None:
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

    def test_p3_not_in_matching(self) -> None:
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

    def test_non_induced_allows_chords(self) -> None:
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

    def test_rejects_pattern_larger_than_host(self) -> None:
        import pytest

        from jacobian.math.graphs.morphisms._models import (
            SubgraphPatternFindRequest,
        )

        pat = self._g(["x", "y", "z"], [["x", "y"], ["y", "z"]])
        host = self._g(["a", "b"], [["a", "b"]])
        with pytest.raises(ValueError):
            SubgraphPatternFindRequest(pattern=pat, host=host)

    def test_composes_with_canonical_graph(self) -> None:
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

    def test_request_admission_charges_the_kernel_pass(self) -> None:
        from jacobian.math.graphs.morphisms._models import (
            MAX_CYCLE_SEARCH_PATHS,
            SubgraphPatternFindRequest,
        )

        # P(11, 8) = 6,652,800 assignments fits in the one kernel pass. Result
        # parsing does not replay the backend search.
        pat = self._g(
            [f"x{i}" for i in range(8)], [[f"x{i}", f"x{i + 1}"] for i in range(7)]
        )
        host_labels = [f"h{i:02d}" for i in range(11)]
        host = self._g(host_labels, [[f"h{i:02d}", f"h{i + 1:02d}"] for i in range(10)])
        assert MAX_CYCLE_SEARCH_PATHS > 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4
        assert SubgraphPatternFindRequest(pattern=pat, host=host).pattern == pat

    def test_request_admission_reserves_output_headroom_for_source_echo(self) -> None:
        import pytest

        from jacobian.math.graphs.morphisms._models import FixedLengthCycleRequest

        # An edgeless 20-vertex graph with one multi-megabyte NFC label fits
        # the canonical input limit, but the result echoes the graph and adds
        # the envelope; admission must reject before any search runs.
        huge = "v" * (6 * 1024 * 1024)
        labels = [huge] + [f"w{i}" for i in range(19)]
        g = self._g(labels, [])
        with pytest.raises(ValueError):
            FixedLengthCycleRequest(graph=g, length=3)

    def test_forged_negative_decision_is_rejected_by_explicit_verifier(self) -> None:
        from jacobian.math.graphs.morphisms._models import SubgraphPatternFindResult
        from jacobian.math.graphs.morphisms._operations import (
            verify_subgraph_pattern_find_result,
        )

        pat = self._g(["x", "y", "z"], [["x", "y"], ["x", "z"], ["y", "z"]])
        host = self._g(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
        forged = SubgraphPatternFindResult(
            pattern=pat, host=host, decision="DOES_NOT_EXIST", vertex_map=()
        )
        assert not verify_subgraph_pattern_find_result(forged)

    def test_negative_decision_is_structural_inside_request_domain(self) -> None:
        import pytest

        from jacobian.math.graphs.morphisms._models import (
            MAX_CYCLE_SEARCH_PATHS,
            MORPHISM_MAX_VERTICES,
            SubgraphPatternFindRequest,
            SubgraphPatternFindResult,
        )
        from jacobian.math.graphs.morphisms._operations import (
            compute_subgraph_pattern_find,
            verify_subgraph_pattern_find_result,
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
        assert verify_subgraph_pattern_find_result(revalidated)

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
        with pytest.raises(ValueError):
            SubgraphPatternFindResult(
                pattern=big_pat, host=big_host, decision="DOES_NOT_EXIST", vertex_map=()
            )
        assert MAX_CYCLE_SEARCH_PATHS > 0


class TestSubgraphPatternFindLabelCost:
    def _g(
        self,
        vertices: list[str] | tuple[str, ...],
        edges: list[list[str]] | tuple[tuple[str, str], ...],
    ) -> SimpleUndirectedGraph:
        return _canonical_graph(vertices, edges)

    def test_long_shared_prefix_labels_decide_correctly(self) -> None:
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
    def _complete(self, n: int) -> SimpleUndirectedGraph:
        from jacobian.math.graphs.values import SimpleUndirectedGraph

        verts = tuple(f"{i:02d}" for i in range(1, n + 1))
        edges = tuple((a, b) for idx, a in enumerate(verts) for b in verts[idx + 1 :])
        return SimpleUndirectedGraph(vertices=verts, edges=edges)

    def test_internal_backtracking_nodes_are_charged_to_the_budget(self) -> None:
        """K10 into K10-minus-an-edge cannot return a free negative.

        A failed search visits 1,863,219 partial mappings and scans all ten
        host candidates at each one (~18.6M candidate checks in one pass).
        The kernel charges those candidate checks to the work budget and
        reports the typed non-conclusion instead of a
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
        # The typed non-conclusion round-trips without a backend replay.
        type(result).model_validate(result.model_dump())

        from jacobian.math.graphs.morphisms._operations import (
            verify_subgraph_pattern_find_result,
        )

        forged = type(result)(
            pattern=request.pattern,
            host=request.host,
            decision="DOES_NOT_EXIST",
            vertex_map=(),
        )
        assert not verify_subgraph_pattern_find_result(forged)
