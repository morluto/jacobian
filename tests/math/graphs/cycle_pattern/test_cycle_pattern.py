"""Tests for graph cycle and subgraph-pattern operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.cycle_pattern._models import (
    FixedLengthCycleRequest,
    FixedLengthCycleResult,
    SubgraphEmbedding,
    SubgraphPatternRequest,
    SubgraphPatternResult,
)
from jacobian.math.graphs.cycle_pattern._operations import (
    decide_fixed_length_cycle,
    find_subgraph_pattern,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(names: list[str], edges: list[tuple[str, str]]) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph.model_validate({"vertices": names, "edges": edges})


class TestFixedLengthCycle:
    """Tests for ``graph.cycle.fixed_length.decide``."""

    def test_triangle(self):
        g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")])
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        assert result.exists
        assert result.cycle == ("a", "b", "c")

    def test_four_cycle(self):
        g = _graph(
            ["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")]
        )
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=4))
        assert result.exists
        assert result.cycle == ("a", "b", "c", "d")

    def test_no_cycle_in_path(self):
        g = _graph(["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "d")])
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=4))
        assert not result.exists

    def test_no_cycle_in_tree(self):
        g = _graph(
            ["a", "b", "c", "d", "e"],
            [("a", "b"), ("b", "c"), ("b", "d"), ("d", "e")],
        )
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        assert not result.exists

    def test_cycle_with_chord(self):
        g = _graph(
            ["a", "b", "c", "d", "e"],
            [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("a", "e"), ("a", "c")],
        )
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=4))
        assert result.exists
        assert len(result.cycle) == 4

    def test_pentagon_has_5_cycle(self):
        g = _graph(
            ["a", "b", "c", "d", "e"],
            [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("a", "e")],
        )
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=5))
        assert result.exists
        assert result.cycle == ("a", "b", "c", "d", "e")

    def test_pentagon_no_3_cycle(self):
        g = _graph(
            ["a", "b", "c", "d", "e"],
            [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("a", "e")],
        )
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        assert not result.exists

    def test_isolated_vertex(self):
        g = _graph(["a", "b", "c", "z"], [("a", "b"), ("b", "c"), ("a", "c")])
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        assert result.exists
        assert result.cycle == ("a", "b", "c")

    def test_self_loop_rejected(self):
        with pytest.raises(ValidationError):
            _graph(["a", "b"], [("a", "a")])

    def test_duplicate_edge_rejected(self):
        with pytest.raises(ValidationError):
            _graph(["a", "b"], [("a", "b"), ("b", "a")])

    def test_length_exceeds_vertex_count(self):
        g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c")])
        with pytest.raises(ValueError, match="cycle length"):
            FixedLengthCycleRequest(graph=g, length=4)

    def test_oversized_graph_rejected(self):
        """The domain caps graphs at 64 vertices for a declared work bound."""
        from jacobian.math.graphs.cycle_pattern._models import MAX_CYCLE_GRAPH_ORDER

        names = [f"v{i:03d}" for i in range(MAX_CYCLE_GRAPH_ORDER + 1)]
        with pytest.raises(ValidationError, match="64 vertices"):
            FixedLengthCycleRequest(graph=_graph(names, []), length=3)


class TestDenseAndLongAdmission:
    """Bounded requests are admitted by the work budget, not fixed caps."""

    def test_bipartite_triangle_free_decided_exactly(self):
        """A complete bipartite host is triangle-free: exact negative, not
        rejected admission, despite exceeding nothing in the budget."""
        left = [f"L{i}" for i in range(3)]
        right = [f"R{i}" for i in range(3)]
        edges = [(lf, rg) for lf in left for rg in right]
        host = _graph(left + right, sorted(edges))
        assert len(host.edges) == 9
        result = decide_fixed_length_cycle(
            FixedLengthCycleRequest(graph=host, length=3)
        )
        assert result.exists is False

    def test_dense_complete_graph_above_old_cap_admitted(self):
        """K_33-style dense graphs beyond the old 512-edge ceiling are
        admitted; the reviewer's example finds its witness immediately."""
        left = [f"L{i:02d}" for i in range(24)]
        right = [f"R{i:02d}" for i in range(24)]
        edges = [(lf, rg) for lf in left for rg in right]
        host = _graph(left + right, edges)
        assert len(host.edges) == 576
        result = decide_fixed_length_cycle(
            FixedLengthCycleRequest(graph=host, length=3)
        )
        assert result.exists is False

    def test_complete_graph_on_33_vertices_finds_triangle(self):
        """The complete graph on 33 vertices has 528 edges (> 512) and its
        triangle witness appears within the first few recursion nodes."""
        names = [f"k{i:02d}" for i in range(33)]
        edges = [(names[i], names[j]) for i in range(33) for j in range(i + 1, 33)]
        host = _graph(names, edges)
        assert len(host.edges) == 528
        result = decide_fixed_length_cycle(
            FixedLengthCycleRequest(graph=host, length=3)
        )
        assert result.exists
        assert len(result.cycle) == 3

    def test_cycle_longer_than_twenty_admitted(self):
        """A 21-cycle request runs within the node budget and finds it."""
        names = [f"w{i:02d}" for i in range(21)]
        edges = [tuple(sorted((names[i], names[(i + 1) % 21]))) for i in range(21)]
        g = _graph(names, edges)
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=21))
        assert result.exists
        assert len(result.cycle) == 21


class TestSubgraphPattern:
    """Tests for ``graph.subgraph.pattern.find``."""

    def test_path_in_pentagon(self):
        host = _graph(
            ["a", "b", "c", "d", "e"],
            [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("a", "e"), ("a", "c")],
        )
        pattern = _graph(["p0", "p1", "p2"], [("p0", "p1"), ("p1", "p2")])
        result = find_subgraph_pattern(
            SubgraphPatternRequest(host=host, pattern=pattern)
        )
        assert result.exists
        mapping = dict(result.embedding.mapping)
        assert len(set(mapping.values())) == 3

    def test_triangle_not_in_single_edge(self):
        host = _graph(["a", "b"], [("a", "b")])
        pattern = _graph(["u", "v", "w"], [("u", "v"), ("v", "w"), ("u", "w")])
        with pytest.raises(ValueError, match="pattern vertex count"):
            SubgraphPatternRequest(host=host, pattern=pattern)

    def test_exact_match(self):
        host = _graph(["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")])
        pattern = _graph(["x", "y", "z"], [("x", "y"), ("y", "z"), ("x", "z")])
        result = find_subgraph_pattern(
            SubgraphPatternRequest(host=host, pattern=pattern)
        )
        assert result.exists
        assert result.embedding.mapping == (("x", "a"), ("y", "b"), ("z", "c"))

    def test_single_vertex_pattern(self):
        host = _graph(["a", "b"], [("a", "b")])
        pattern = _graph(["only"], [])
        result = find_subgraph_pattern(
            SubgraphPatternRequest(host=host, pattern=pattern)
        )
        assert result.exists
        assert result.embedding.mapping == (("only", "a"),)

    def test_pattern_too_large(self):
        host = _graph(["a", "b"], [("a", "b")])
        pattern = _graph(["u", "v", "w"], [("u", "v")])
        with pytest.raises(ValueError, match="pattern vertex count"):
            SubgraphPatternRequest(host=host, pattern=pattern)

    def test_clique_in_larger_clique(self):
        host = _graph(
            ["a", "b", "c", "d"],
            [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")],
        )
        pattern = _graph(["u", "v", "w"], [("u", "v"), ("v", "w"), ("u", "w")])
        result = find_subgraph_pattern(
            SubgraphPatternRequest(host=host, pattern=pattern)
        )
        assert result.exists

    def test_triangle_not_in_cycle_six(self):
        host = _graph(
            ["a", "b", "c", "d", "e", "f"],
            [
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("d", "e"),
                ("e", "f"),
                ("a", "f"),
            ],
        )
        pattern = _graph(["u", "v", "w"], [("u", "v"), ("v", "w"), ("u", "w")])
        result = find_subgraph_pattern(
            SubgraphPatternRequest(host=host, pattern=pattern)
        )
        assert not result.exists


class TestBoundedSearchAndWitnessBinding:
    """Searches carry a deterministic work bound; witnesses replay sources."""

    def test_cycle_witness_replays_against_source_edges(self):
        g = _graph(
            ["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")]
        )
        request = FixedLengthCycleRequest(graph=g, length=4)
        result = decide_fixed_length_cycle(request)
        assert result.exists and result.cycle is not None
        # A witness whose edges do not exist in the source graph is rejected.
        with pytest.raises(ValidationError, match="replay"):
            FixedLengthCycleResult(
                graph=request.graph,
                length=4,
                exists=True,
                cycle=("a", "b", "d", "c"),
            )

    def test_embedding_preserves_pattern_edges_or_rejected(self):
        host = _graph(
            ["a", "b", "c", "d"],
            [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"), ("a", "c")],
        )
        pattern = _graph(["u", "v", "w"], [("u", "v"), ("v", "w"), ("u", "w")])
        result = find_subgraph_pattern(
            SubgraphPatternRequest(host=host, pattern=pattern)
        )
        assert result.exists
        with pytest.raises(ValidationError):
            SubgraphPatternResult(
                host_graph=host,
                pattern_graph=pattern,
                exists=True,
                embedding=SubgraphEmbedding(
                    mapping=(("u", "a"), ("v", "b"), ("w", "d"))
                ),
            )

    def test_budget_exceeded_is_typed_not_hang(self, monkeypatch):
        """A triangle-free dense graph cannot decide k=3 within a tiny budget."""
        from jacobian.math.graphs.cycle_pattern import _operations as ops

        left = [f"L{i:02d}" for i in range(16)]
        right = [f"R{i:02d}" for i in range(16)]
        graph = _graph(left + right, [(lf, rg) for lf in left for rg in right])
        monkeypatch.setattr(ops, "MAX_SEARCH_NODES", 100)
        request = FixedLengthCycleRequest(graph=graph, length=3)
        result = ops.decide_fixed_length_cycle(request)
        assert result.outcome == "SEARCH_BUDGET_EXCEEDED"
        assert result.exists is None and result.cycle is None

    def test_negative_embedding_replay_stays_within_search_budget(self):
        """An empty 64-vertex host against a 20-vertex one-edge pattern is
        rejected by degree pruning; validating the negative result must
        replay the same bounded search instead of enumerating P(64,20)
        injective mappings."""
        import time

        host = _graph([f"h{i:02d}" for i in range(64)], [])
        pattern = _graph([f"p{i:02d}" for i in range(20)], [("p00", "p01")])
        start = time.monotonic()
        result = find_subgraph_pattern(
            SubgraphPatternRequest(host=host, pattern=pattern)
        )
        elapsed = time.monotonic() - start
        assert result.exists is False
        assert elapsed < 5.0

    def test_forged_negative_embedding_is_rejected(self):
        """A host containing the pattern cannot validate exists=False."""
        host = _graph(
            ["a", "b", "c", "d"],
            [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d"), ("a", "c")],
        )
        pattern = _graph(["u", "v", "w"], [("u", "v"), ("v", "w"), ("u", "w")])
        with pytest.raises(ValidationError, match="contradicts"):
            SubgraphPatternResult(
                host_graph=host,
                pattern_graph=pattern,
                exists=False,
            )

    def test_forged_negative_cycle_is_rejected(self):
        """A graph containing the requested cycle cannot validate exists=False."""
        square = _graph(
            ["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")]
        )
        with pytest.raises(ValidationError, match="contradicts"):
            FixedLengthCycleResult(graph=square, length=4, exists=False)


class TestEmbeddingFunctionContract:
    """An authored mapping must be an injective function on distinct names."""

    def test_duplicate_domain_vertices_rejected(self):
        """Duplicate domain entries silently vanish under dict conversion."""
        host = _graph(["h1", "h2"], [("h1", "h2")])
        pattern = _graph(["u", "v"], [])
        with pytest.raises(ValidationError, match="domain vertices must be distinct"):
            SubgraphPatternResult(
                host_graph=host,
                pattern_graph=pattern,
                exists=True,
                embedding=SubgraphEmbedding(mapping=(("u", "u"), ("u", "h1"))),
            )

    def test_codomain_bounds_enforced_by_name(self):
        host = _graph(["h1", "h2"], [])
        pattern = _graph(["u", "v"], [])
        request = SubgraphPatternRequest(host=host, pattern=pattern)
        with pytest.raises(ValidationError, match=r"host graph|lie in the host"):
            SubgraphPatternResult(
                host_graph=request.host,
                pattern_graph=request.pattern,
                exists=True,
                embedding=SubgraphEmbedding(mapping=(("u", "zz"), ("v", "yy"))),
            )


class TestCanonicalValueComposition:
    """Serialized producer output composes into consumers unchanged."""

    def test_serialized_request_and_result_round_trip(self):
        g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")])
        payload = FixedLengthCycleRequest(graph=g, length=3).model_dump()
        relayed = FixedLengthCycleRequest.model_validate(payload)
        assert relayed.graph == g
        result = decide_fixed_length_cycle(relayed)
        revalidated = type(result).model_validate(result.model_dump())
        assert revalidated.cycle == result.cycle

    def test_producer_output_feeds_other_graph_consumers(self):
        """The retained graph value enters another canonical consumer unchanged."""
        g = _graph(["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")])
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        # The same canonical value object round-trips through the cycle
        # operation and remains valid as a SimpleUndirectedGraph elsewhere.
        assert result.graph == g
        assert SimpleUndirectedGraph.model_validate(g.model_dump()) == g
