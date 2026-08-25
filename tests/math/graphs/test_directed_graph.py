"""Tests for directed graph reachability, SCC, condensation, and acyclic order."""

from __future__ import annotations

import networkx as nx
import pytest
from pydantic import ValidationError

from jacobian.math.graphs.directed._models import (
    AcyclicOrderRequest,
    AcyclicOrderResult,
    CondensationRequest,
    CondensationResult,
    ReachabilityRequest,
    ReachabilityResult,
    StronglyConnectedComponentsRequest,
    StronglyConnectedComponentsResult,
)
from jacobian.math.graphs.directed._operations import (
    compute_acyclic_order,
    compute_condensation,
    compute_reachability,
    compute_strongly_connected_components,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reachability(graph: dict, source: int) -> ReachabilityResult:
    return compute_reachability(
        ReachabilityRequest.model_validate({"graph": graph, "source": source})
    )


def _scc(graph: dict) -> StronglyConnectedComponentsResult:
    return compute_strongly_connected_components(
        StronglyConnectedComponentsRequest.model_validate({"graph": graph})
    )


def _condensation(graph: dict) -> CondensationResult:
    return compute_condensation(CondensationRequest.model_validate({"graph": graph}))


def _acyclic_order(graph: dict) -> AcyclicOrderResult:
    return compute_acyclic_order(AcyclicOrderRequest.model_validate({"graph": graph}))


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------


class TestReachability:
    def test_all_vertices_reachable_in_chain(self) -> None:
        result = _reachability(
            {"vertex_count": 4, "edges": [[0, 1], [1, 2], [2, 3]]},
            0,
        )
        assert result.reachable == (0, 1, 2, 3)
        assert result.unreachable == ()

    def test_unreachable_vertices(self) -> None:
        """Vertices in a separate component should be unreachable."""
        result = _reachability(
            {"vertex_count": 4, "edges": [[0, 1], [2, 3]]},
            0,
        )
        assert result.reachable == (0, 1)
        assert result.unreachable == (2, 3)

    def test_source_always_reachable(self) -> None:
        """The source vertex is reachable even with no outgoing edges."""
        result = _reachability(
            {"vertex_count": 3, "edges": [[0, 1], [1, 2]]},
            2,
        )
        assert result.reachable == (2,)
        assert result.unreachable == (0, 1)

    def test_directed_edges_only_followed_in_one_direction(self) -> None:
        """A directed edge 1 -> 0 does not make 1 reachable from 0."""
        result = _reachability(
            {"vertex_count": 2, "edges": [[1, 0]]},
            0,
        )
        assert result.reachable == (0,)
        assert result.unreachable == (1,)

    def test_source_field_in_result(self) -> None:
        result = _reachability(
            {"vertex_count": 3, "edges": [[0, 1], [1, 2]]},
            1,
        )
        assert result.source == 1
        assert result.reachable == (1, 2)

    def test_edgeless_graph_reaches_only_the_source(self) -> None:
        """An edgeless directed graph is a valid degenerate input."""
        result = _reachability(
            {"vertex_count": 2, "edges": []},
            1,
        )
        assert result.reachable == (1,)
        assert result.unreachable == (0,)

    def test_reachability_above_the_shared_vertex_cap(self) -> None:
        """Vertex counts above 64 stay admitted with sources past index 63."""
        result = _reachability(
            {"vertex_count": 100, "edges": [[63, 64], [64, 99]]},
            63,
        )
        assert result.reachable == (63, 64, 99)
        assert len(result.unreachable) == 97


class TestReachabilityContract:
    def test_rejects_self_loop(self) -> None:
        with pytest.raises(ValidationError):
            ReachabilityRequest.model_validate(
                {"graph": {"vertex_count": 2, "edges": [[0, 0], [0, 1]]}, "source": 0}
            )

    def test_rejects_out_of_range_edge_vertex(self) -> None:
        with pytest.raises(ValidationError):
            ReachabilityRequest.model_validate(
                {"graph": {"vertex_count": 2, "edges": [[0, 3]]}, "source": 0}
            )

    def test_rejects_duplicate_edges(self) -> None:
        with pytest.raises(ValidationError):
            ReachabilityRequest.model_validate(
                {"graph": {"vertex_count": 3, "edges": [[0, 1], [0, 1]]}, "source": 0}
            )

    def test_rejects_out_of_range_source(self) -> None:
        with pytest.raises(ValidationError):
            ReachabilityRequest.model_validate(
                {"graph": {"vertex_count": 2, "edges": [[0, 1]]}, "source": 5}
            )

    def test_rejects_vertex_count_above_the_conservative_fallback(self) -> None:
        with pytest.raises(ValidationError):
            ReachabilityRequest.model_validate(
                {"graph": {"vertex_count": 257, "edges": []}, "source": 0}
            )


# ---------------------------------------------------------------------------
# Strongly connected components
# ---------------------------------------------------------------------------


class TestStronglyConnectedComponents:
    def test_single_cycle_is_one_component(self) -> None:
        """A single cycle covers all vertices in one SCC."""
        result = _scc(
            {"vertex_count": 3, "edges": [[0, 1], [1, 2], [2, 0]]},
        )
        assert result.component_count == 1
        assert result.components == ((0, 1, 2),)

    def test_dag_has_singletons(self) -> None:
        """A DAG has one SCC per vertex."""
        result = _scc(
            {"vertex_count": 3, "edges": [[0, 1], [1, 2]]},
        )
        assert result.component_count == 3
        # Each component is a singleton
        singleton_sizes = {len(c) for c in result.components}
        assert singleton_sizes == {1}

    def test_mixed_cycle_and_singletons(self) -> None:
        result = _scc(
            {"vertex_count": 4, "edges": [[0, 1], [1, 2], [2, 0], [2, 3]]},
        )
        assert result.component_count == 2
        # The non-trivial SCC {0, 1, 2}
        big = next(c for c in result.components if len(c) > 1)
        assert set(big) == {0, 1, 2}
        # The singleton {3}
        singletons = [c for c in result.components if len(c) == 1]
        assert singletons == [(3,)]

    def test_two_separate_cycles(self) -> None:
        result = _scc(
            {"vertex_count": 4, "edges": [[0, 1], [1, 0], [2, 3], [3, 2]]},
        )
        assert result.component_count == 2
        assert {len(c) for c in result.components} == {2}


# ---------------------------------------------------------------------------
# Condensation
# ---------------------------------------------------------------------------


class TestCondensation:
    def test_condensation_of_dag_is_itself(self) -> None:
        """A DAG's condensation has one vertex per original vertex. The
        condensation edges, mapped back through the components, must equal
        the original edge set."""
        edges = [[0, 1], [0, 2], [1, 3], [2, 3]]
        graph = {"vertex_count": 4, "edges": edges}
        result = _condensation(graph)
        assert result.vertex_count == 4
        # Each component is a singleton.
        assert all(len(c) == 1 for c in result.components)
        assert len(result.components) == 4
        # Map each condensation vertex to its single original vertex.
        vertex_of = {i: c[0] for i, c in enumerate(result.components)}
        reconstructed = {
            (vertex_of[e.source], vertex_of[e.target]) for e in result.edges
        }
        assert reconstructed == {tuple(e) for e in edges}

    def test_condensation_is_acyclic(self) -> None:
        """The condensation of any directed graph is always a DAG."""
        # Graph with a cycle that reaches a sink.
        graph = {"vertex_count": 4, "edges": [[0, 1], [1, 2], [2, 0], [2, 3]]}
        result = _condensation(graph)
        # Build the condensation as a NetworkX graph and verify acyclicity.
        cond = nx.DiGraph()
        cond.add_nodes_from(range(result.vertex_count))
        cond.add_edges_from((e.source, e.target) for e in result.edges)
        assert nx.is_directed_acyclic_graph(cond)

    def test_condensation_collapse_cycle_into_single_vertex(self) -> None:
        """A single cycle should collapse to one condensation vertex."""
        graph = {"vertex_count": 3, "edges": [[0, 1], [1, 2], [2, 0]]}
        result = _condensation(graph)
        assert result.vertex_count == 1
        assert result.components == ((0, 1, 2),)
        assert result.edges == ()

    def test_condensation_edge_from_cycle_to_sink(self) -> None:
        """Cycle {0,1,2} -> sink {3} should produce one condensation edge."""
        graph = {"vertex_count": 4, "edges": [[0, 1], [1, 2], [2, 0], [2, 3]]}
        result = _condensation(graph)
        assert result.vertex_count == 2
        # Identify which component index is the cycle and which is the sink.
        cycle_idx = next(
            i for i, c in enumerate(result.components) if set(c) == {0, 1, 2}
        )
        sink_idx = next(i for i, c in enumerate(result.components) if set(c) == {3})
        # The edge should go from the cycle to the sink.
        assert (cycle_idx, sink_idx) in {(e.source, e.target) for e in result.edges}


# ---------------------------------------------------------------------------
# Acyclic order (topological sort)
# ---------------------------------------------------------------------------


class TestAcyclicOrder:
    def test_valid_topological_order_for_dag(self) -> None:
        graph = {"vertex_count": 4, "edges": [[0, 1], [0, 2], [1, 3], [2, 3]]}
        result = _acyclic_order(graph)
        assert result.acyclic
        order = result.order
        # Every vertex must appear exactly once.
        assert sorted(order) == [0, 1, 2, 3]
        # The order must respect all edges.
        position = {v: i for i, v in enumerate(order)}
        assert all(position[u] < position[v] for u, v in graph["edges"])

    def test_chain_topological_order(self) -> None:
        graph = {"vertex_count": 3, "edges": [[0, 1], [1, 2]]}
        result = _acyclic_order(graph)
        assert result.acyclic
        # The only valid topological order is (0, 1, 2).
        assert result.order == (0, 1, 2)

    def test_cyclic_graph_raises(self) -> None:
        graph = {"vertex_count": 3, "edges": [[0, 1], [1, 2], [2, 0]]}
        result = _acyclic_order(graph)
        assert not result.acyclic
        assert result.order == ()

    def test_two_node_cycle_raises(self) -> None:
        graph = {"vertex_count": 2, "edges": [[0, 1], [1, 0]]}
        result = _acyclic_order(graph)
        assert not result.acyclic
        assert result.order == ()

    def test_single_edge_dag(self) -> None:
        result = _acyclic_order({"vertex_count": 2, "edges": [[0, 1]]})
        assert result.acyclic
        assert result.order == (0, 1)


# ---------------------------------------------------------------------------
# Cross-consistency
# ---------------------------------------------------------------------------


class TestCrossConsistency:
    def test_scc_count_matches_condensation_vertex_count(self) -> None:
        """The condensation vertex count equals the SCC component count."""
        graph = {"vertex_count": 4, "edges": [[0, 1], [1, 2], [2, 0], [2, 3]]}
        scc_result = _scc(graph)
        cond_result = _condensation(graph)
        assert scc_result.component_count == cond_result.vertex_count

    def test_condensation_components_match_scc_components(self) -> None:
        """The condensation's components should match the SCC components."""
        graph = {"vertex_count": 4, "edges": [[0, 1], [1, 2], [2, 0], [2, 3]]}
        scc_result = _scc(graph)
        cond_result = _condensation(graph)
        scc_as_sets = {frozenset(c) for c in scc_result.components}
        cond_as_sets = {frozenset(c) for c in cond_result.components}
        assert scc_as_sets == cond_as_sets
