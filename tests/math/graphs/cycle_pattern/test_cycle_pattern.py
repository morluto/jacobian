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
    UndirectedGraph,
)
from jacobian.math.graphs.cycle_pattern._operations import (
    decide_fixed_length_cycle,
    find_subgraph_pattern,
)


class TestFixedLengthCycle:
    """Tests for ``graph.cycle.fixed_length.decide``."""

    def test_triangle(self):
        g = UndirectedGraph(vertex_count=3, edges=((0, 1), (1, 2), (0, 2)))
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        assert result.exists
        assert result.cycle == (0, 1, 2)

    def test_four_cycle(self):
        g = UndirectedGraph(vertex_count=4, edges=((0, 1), (1, 2), (2, 3), (3, 0)))
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=4))
        assert result.exists
        assert result.cycle == (0, 1, 2, 3)

    def test_no_cycle_in_path(self):
        g = UndirectedGraph(vertex_count=4, edges=((0, 1), (1, 2), (2, 3)))
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=4))
        assert not result.exists

    def test_no_cycle_in_tree(self):
        g = UndirectedGraph(
            vertex_count=5,
            edges=((0, 1), (1, 2), (1, 3), (3, 4)),
        )
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        assert not result.exists

    def test_cycle_with_chord(self):
        g = UndirectedGraph(
            vertex_count=5,
            edges=((0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)),
        )
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=4))
        assert result.exists
        assert len(result.cycle) == 4

    def test_pentagon_has_5_cycle(self):
        g = UndirectedGraph(
            vertex_count=5,
            edges=((0, 1), (1, 2), (2, 3), (3, 4), (4, 0)),
        )
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=5))
        assert result.exists
        assert result.cycle == (0, 1, 2, 3, 4)

    def test_pentagon_no_3_cycle(self):
        g = UndirectedGraph(
            vertex_count=5,
            edges=((0, 1), (1, 2), (2, 3), (3, 4), (4, 0)),
        )
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        assert not result.exists

    def test_isolated_vertex(self):
        g = UndirectedGraph(vertex_count=4, edges=((0, 1), (1, 2), (2, 0)))
        result = decide_fixed_length_cycle(FixedLengthCycleRequest(graph=g, length=3))
        assert result.exists
        assert result.cycle == (0, 1, 2)

    def test_self_loop_rejected(self):
        with pytest.raises(ValueError, match="self-loops"):
            UndirectedGraph(vertex_count=3, edges=((0, 0), (1, 2)))

    def test_duplicate_edge_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            UndirectedGraph(vertex_count=3, edges=((0, 1), (1, 0)))

    def test_length_exceeds_vertex_count(self):
        g = UndirectedGraph(vertex_count=3, edges=((0, 1), (1, 2)))
        with pytest.raises(ValueError, match="cycle length"):
            FixedLengthCycleRequest(graph=g, length=4)


class TestSubgraphPattern:
    """Tests for ``graph.subgraph.pattern.find``."""

    def test_path_in_pentagon(self):
        host = UndirectedGraph(
            vertex_count=5,
            edges=((0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)),
        )
        pattern = UndirectedGraph(vertex_count=3, edges=((0, 1), (1, 2)))
        result = find_subgraph_pattern(SubgraphPatternRequest(host=host, pattern=pattern))
        assert result.exists
        mapping = dict(result.embedding.mapping)
        assert mapping[0] != mapping[1]
        assert mapping[1] != mapping[2]
        assert mapping[0] != mapping[2]

    def test_triangle_not_in_edge(self):
        host = UndirectedGraph(vertex_count=3, edges=((0, 1),))
        pattern = UndirectedGraph(
            vertex_count=3,
            edges=((0, 1), (1, 2), (0, 2)),
        )
        result = find_subgraph_pattern(SubgraphPatternRequest(host=host, pattern=pattern))
        assert not result.exists

    def test_exact_match(self):
        host = UndirectedGraph(
            vertex_count=3,
            edges=((0, 1), (1, 2), (0, 2)),
        )
        pattern = UndirectedGraph(
            vertex_count=3,
            edges=((0, 1), (1, 2), (0, 2)),
        )
        result = find_subgraph_pattern(SubgraphPatternRequest(host=host, pattern=pattern))
        assert result.exists
        assert result.embedding.mapping == ((0, 0), (1, 1), (2, 2))

    def test_single_vertex_pattern(self):
        host = UndirectedGraph(vertex_count=3, edges=((0, 1),))
        pattern = UndirectedGraph(vertex_count=1, edges=())
        result = find_subgraph_pattern(SubgraphPatternRequest(host=host, pattern=pattern))
        assert result.exists
        assert result.embedding.mapping == ((0, 0),)

    def test_pattern_too_large(self):
        host = UndirectedGraph(vertex_count=2, edges=((0, 1),))
        pattern = UndirectedGraph(vertex_count=3, edges=((0, 1),))
        with pytest.raises(ValueError, match="pattern vertex count"):
            SubgraphPatternRequest(host=host, pattern=pattern)

    def test_clique_in_larger_clique(self):
        host = UndirectedGraph(
            vertex_count=4,
            edges=((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        )
        pattern = UndirectedGraph(
            vertex_count=3,
            edges=((0, 1), (1, 2), (0, 2)),
        )
        result = find_subgraph_pattern(SubgraphPatternRequest(host=host, pattern=pattern))
        assert result.exists

    def test_triangle_not_in_cycle_six(self):
        host = UndirectedGraph(
            vertex_count=6,
            edges=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)),
        )
        pattern = UndirectedGraph(
            vertex_count=3,
            edges=((0, 1), (1, 2), (0, 2)),
        )
        result = find_subgraph_pattern(SubgraphPatternRequest(host=host, pattern=pattern))
        assert not result.exists


class TestBoundedSearchAndWitnessBinding:
    """Searches carry a deterministic work bound; witnesses replay sources."""

    def _graph(self, n, edges):
        return {"vertex_count": n, "edges": [list(e) for e in edges]}

    def test_cycle_witness_replays_against_source_edges(self):
        request = FixedLengthCycleRequest(
            graph=UndirectedGraph.model_validate(self._graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])),
            length=4,
        )
        result = decide_fixed_length_cycle(request)
        assert result.exists and result.cycle is not None
        # A witness whose edges do not exist in the source graph is rejected.
        with pytest.raises(ValidationError, match="replay"):
            FixedLengthCycleResult(
                graph=request.graph,
                vertex_count=4,
                length=4,
                exists=True,
                cycle=(0, 1, 3, 2),
            )

    def test_embedding_preserves_pattern_edges_or_rejected(self):
        host = UndirectedGraph.model_validate(self._graph(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]))
        pattern = UndirectedGraph.model_validate(self._graph(3, [(0, 1), (1, 2), (2, 0)]))
        result = find_subgraph_pattern(SubgraphPatternRequest(host=host, pattern=pattern))
        assert result.exists
        with pytest.raises(ValidationError):
            SubgraphPatternResult(
                host_graph=host,
                pattern_graph=pattern,
                host_vertex_count=4,
                pattern_vertex_count=3,
                exists=True,
                embedding=SubgraphEmbedding(mapping=((0, 0), (1, 1), (2, 3))),
            )

    def test_budget_exceeded_is_typed_not_hang(self, monkeypatch):
        """A triangle-free dense graph cannot decide k=3 within a tiny budget."""
        from jacobian.math.graphs.cycle_pattern import _operations as ops

        graph = UndirectedGraph.model_validate(
            self._graph(32, [(a, b) for a in range(16) for b in range(16, 32)])
        )
        monkeypatch.setattr(ops, "MAX_SEARCH_NODES", 100)
        request = FixedLengthCycleRequest(graph=graph, length=3)
        result = ops.decide_fixed_length_cycle(request)
        assert result.outcome == "SEARCH_BUDGET_EXCEEDED"
        assert result.exists is None and result.cycle is None
