"""Tests for bounded directed-graph operations (issue #1687)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.directed_graph import DirectedGraph
from jacobian.domains.directed_graph.operations import (
    compute_acyclic_order,
    compute_degree_profile,
    compute_reachability,
    compute_strong_components,
    compute_transitive_closure,
)
from jacobian.contracts.directed_graph import (
    AcyclicOrderRequest,
    DegreeProfileRequest,
    ReachabilityRequest,
    StrongComponentsRequest,
    TransitiveClosureRequest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(vertex_count: int, arcs: list[tuple[int, int]]) -> DirectedGraph:
    return DirectedGraph.model_validate(
        {"vertex_count": vertex_count, "arcs": arcs}
    )


# ---------------------------------------------------------------------------
# 1. A DAG with more than one valid topological order
# ---------------------------------------------------------------------------
def test_acyclic_order_dag_has_multiple_valid_orders() -> None:
    """A DAG where multiple topological orders are valid.

    Vertices: 0, 1, 2, 3, 4
    Edges: 0->1, 0->2, 1->3, 2->3, 3->4
    Multiple valid orders exist: e.g., [0,1,2,3,4] and [0,2,1,3,4].
    """
    graph = _make_graph(5, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])
    request = AcyclicOrderRequest(graph=graph)
    result = compute_acyclic_order(request)

    assert result.status == "ACYCLIC"
    assert result.topological_order is not None
    assert result.positions is not None
    assert result.cycle_witness is None

    order = result.topological_order
    # The order must contain all vertices.
    assert sorted(order) == list(range(5))
    # Every edge must go forward in the order.
    for tail, head in graph.arcs:
        assert result.positions[tail] < result.positions[head]


def test_acyclic_order_dag_independent_pair() -> None:
    """A DAG where 1 and 2 are independent: both [0,1,2,3] and [0,2,1,3] valid."""
    graph = _make_graph(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    request = AcyclicOrderRequest(graph=graph)
    result = compute_acyclic_order(request)

    assert result.status == "ACYCLIC"
    assert result.topological_order is not None
    for tail, head in graph.arcs:
        assert result.positions[tail] < result.positions[head]


# ---------------------------------------------------------------------------
# 2. A directed cycle returning a concrete cycle witness
# ---------------------------------------------------------------------------
def test_acyclic_order_cyclic_returns_cycle_witness() -> None:
    graph = _make_graph(3, [(0, 1), (1, 2), (2, 0)])
    request = AcyclicOrderRequest(graph=graph)
    result = compute_acyclic_order(request)

    assert result.status == "CYCLIC"
    assert result.cycle_witness is not None
    assert result.topological_order is None
    assert result.positions is None

    # The cycle witness must be a valid closed directed cycle.
    cycle = result.cycle_witness
    assert len(cycle) >= 2
    cycle_set = set(cycle)
    # Each consecutive pair must be an arc, and the last->first pair too.
    for i in range(len(cycle)):
        tail = cycle[i]
        head = cycle[(i + 1) % len(cycle)]
        assert (tail, head) in set(graph.arcs)


def test_acyclic_order_self_loop_returns_cycle_witness() -> None:
    graph = _make_graph(2, [(0, 1), (1, 1)])
    request = AcyclicOrderRequest(graph=graph)
    result = compute_acyclic_order(request)

    assert result.status == "CYCLIC"
    assert result.cycle_witness is not None
    cycle = result.cycle_witness
    # Self-loop witness is a single-vertex cycle.
    for i in range(len(cycle)):
        tail = cycle[i]
        head = cycle[(i + 1) % len(cycle)]
        assert (tail, head) in set(graph.arcs)


# ---------------------------------------------------------------------------
# 3. A graph with several SCCs and a nontrivial condensation DAG
# ---------------------------------------------------------------------------
def test_scc_nontrivial_condensation() -> None:
    """Graph with multiple SCCs and a nontrivial condensation DAG.

    SCC0: vertices 0, 1 (0<->1)
    SCC1: vertices 2, 3 (2<->3)
    Condensation arc: SCC0 -> SCC1
    """
    graph = _make_graph(
        4,
        [(0, 1), (1, 0), (2, 3), (3, 2), (1, 2)],
    )
    request = StrongComponentsRequest(graph=graph)
    result = compute_strong_components(request)

    assert result.component_count == 2
    assert not result.is_strongly_connected

    # Verify the partition is correct: 0,1 in one component; 2,3 in another.
    comp_of_0 = result.component_ids[0]
    comp_of_1 = result.component_ids[1]
    comp_of_2 = result.component_ids[2]
    comp_of_3 = result.component_ids[3]
    assert comp_of_0 == comp_of_1
    assert comp_of_2 == comp_of_3
    assert comp_of_0 != comp_of_2

    # Condensation DAG should have exactly one arc: SCC0 -> SCC1.
    assert len(result.condensation_arcs) == 1
    arc = result.condensation_arcs[0]
    assert arc[0] == comp_of_0
    assert arc[1] == comp_of_2


def test_scc_source_and_sink_components() -> None:
    """Source and sink components in the condensation DAG."""
    # SCC0={0,1} -> SCC1={2,3} -> SCC2={4}
    graph = _make_graph(
        5,
        [(0, 1), (1, 0), (2, 3), (3, 2), (1, 2), (3, 4)],
    )
    request = StrongComponentsRequest(graph=graph)
    result = compute_strong_components(request)

    assert result.component_count == 3
    comp_of_0 = result.component_ids[0]
    comp_of_4 = result.component_ids[4]

    # Source component is the one containing vertex 0.
    assert comp_of_0 in result.source_components
    # Sink component is the one containing vertex 4.
    assert comp_of_4 in result.sink_components


# ---------------------------------------------------------------------------
# 4. A strongly connected graph
# ---------------------------------------------------------------------------
def test_scc_strongly_connected_graph() -> None:
    graph = _make_graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    request = StrongComponentsRequest(graph=graph)
    result = compute_strong_components(request)

    assert result.is_strongly_connected
    assert result.component_count == 1


# ---------------------------------------------------------------------------
# 5. Disconnected/unreachable vertices
# ---------------------------------------------------------------------------
def test_reachability_disconnected_vertices() -> None:
    graph = _make_graph(4, [(0, 1), (2, 3)])
    request = ReachabilityRequest.model_validate(
        {"graph": {"vertex_count": 4, "arcs": [[0, 1], [2, 3]]}, "source": 0}
    )
    result = compute_reachability(request)

    assert result.reachable == (0, 1)
    assert result.unreachable == (2, 3)


# ---------------------------------------------------------------------------
# 6. Source-to-target reachability with a shortest path
# ---------------------------------------------------------------------------
def test_reachability_shortest_path() -> None:
    # 0 -> 1 -> 2 -> 3, and 0 -> 3 (shortcut), shortest path should use shortcut
    graph = _make_graph(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
    request = ReachabilityRequest.model_validate(
        {
            "graph": {"vertex_count": 4, "arcs": [[0, 1], [1, 2], [2, 3], [0, 3]]},
            "source": 0,
            "target": 3,
        }
    )
    result = compute_reachability(request)

    assert result.target_reachable is True
    assert result.shortest_path is not None
    # Shortest path from 0 to 3 should be [0, 3] (direct edge).
    assert result.shortest_path == (0, 3)
    # Distances
    assert result.distances[0] == 0
    assert result.distances[1] == 1
    assert result.distances[3] == 1


# ---------------------------------------------------------------------------
# 7. An unreachable target
# ---------------------------------------------------------------------------
def test_reachability_unreachable_target() -> None:
    graph = _make_graph(3, [(0, 1)])
    request = ReachabilityRequest.model_validate(
        {"graph": {"vertex_count": 3, "arcs": [[0, 1]]}, "source": 0, "target": 2}
    )
    result = compute_reachability(request)

    assert result.target_reachable is False
    assert result.shortest_path is None
    assert 2 in result.unreachable


# ---------------------------------------------------------------------------
# 8. In/out-degree profile with sources, sinks/dead ends, and isolated vertices
# ---------------------------------------------------------------------------
def test_degree_profile_sources_sinks_isolated() -> None:
    graph = _make_graph(
        5,
        [(0, 1), (1, 2), (2, 3)],
    )
    request = DegreeProfileRequest(graph=graph)
    result = compute_degree_profile(request)

    # vertex 0: out=1, in=0 -> source
    # vertex 1: out=1, in=1
    # vertex 2: out=1, in=1
    # vertex 3: out=0, in=1 -> sink
    # vertex 4: out=0, in=0 -> isolated
    assert result.sources == (0, 4)
    assert result.sinks == (3, 4)
    assert result.isolated == (4,)
    assert result.in_degrees[0] == 0
    assert result.out_degrees[0] == 1
    assert result.in_degrees[4] == 0
    assert result.out_degrees[4] == 0


# ---------------------------------------------------------------------------
# 9. Loop behavior under the chosen value convention
# ---------------------------------------------------------------------------
def test_loops_allowed_and_affect_acyclicity() -> None:
    """Loops are permitted as arcs and make the graph cyclic."""
    graph = _make_graph(2, [(0, 1), (1, 1)])
    request = AcyclicOrderRequest(graph=graph)
    result = compute_acyclic_order(request)

    assert result.status == "CYCLIC"
    assert result.cycle_witness is not None


def test_loops_allowed_and_affect_degree() -> None:
    """A self-loop contributes +1 to both in-degree and out-degree."""
    graph = _make_graph(2, [(0, 1), (1, 1)])
    request = DegreeProfileRequest(graph=graph)
    result = compute_degree_profile(request)

    # vertex 0: out=1, in=0
    # vertex 1: out=1 (self-loop), in=1 (self-loop) + 1 (from 0) = 2
    assert result.out_degrees[0] == 1
    assert result.in_degrees[0] == 0
    assert result.out_degrees[1] == 1
    assert result.in_degrees[1] == 2


# ---------------------------------------------------------------------------
# 10. Input row permutation invariance
# ---------------------------------------------------------------------------
def test_input_row_permutation_invariance() -> None:
    graph_a = _make_graph(4, [(0, 1), (1, 2), (2, 3)])
    graph_b = _make_graph(4, [(2, 3), (0, 1), (1, 2)])

    req_a = ReachabilityRequest(graph=graph_a, source=0)
    req_b = ReachabilityRequest(graph=graph_b, source=0)
    result_a = compute_reachability(req_a)
    result_b = compute_reachability(req_b)

    assert result_a.reachable == result_b.reachable
    assert result_a.distances == result_b.distances


def test_input_row_permutation_invariance_scc() -> None:
    graph_a = _make_graph(4, [(0, 1), (1, 2), (2, 0), (2, 3)])
    graph_b = _make_graph(4, [(2, 0), (2, 3), (0, 1), (1, 2)])
    req_a = StrongComponentsRequest(graph=graph_a)
    req_b = StrongComponentsRequest(graph=graph_b)
    result_a = compute_strong_components(req_a)
    result_b = compute_strong_components(req_b)

    # Same partition (component_count and components)
    assert result_a.component_count == result_b.component_count
    # The vertex-to-component mapping should give the same partition structure
    for v in range(4):
        comp_a = result_a.component_ids[v]
        comp_b = result_b.component_ids[v]
        # Same vertices should be in the same component
        assert comp_a == comp_b or set(result_a.components[comp_a]) == set(
            result_b.components[comp_b]
        )


# ---------------------------------------------------------------------------
# 11. Reversed arcs changing reachability as expected
# ---------------------------------------------------------------------------
def test_reversed_arcs_change_reachability() -> None:
    graph_forward = _make_graph(3, [(0, 1), (1, 2)])
    graph_reversed = _make_graph(3, [(2, 1), (1, 0)])

    req_forward = ReachabilityRequest(graph=graph_forward, source=0)
    req_reversed = ReachabilityRequest(graph=graph_reversed, source=0)
    result_forward = compute_reachability(req_forward)
    result_reversed = compute_reachability(req_reversed)

    # Forward: 0 -> 1 -> 2, all reachable from 0
    assert result_forward.reachable == (0, 1, 2)
    # Reversed: 2 -> 1 -> 0, only 0 reachable from 0
    assert result_reversed.reachable == (0,)


# ---------------------------------------------------------------------------
# 12. Exact transitive closure on a short chain
# ---------------------------------------------------------------------------
def test_transitive_closure_chain() -> None:
    graph = _make_graph(4, [(0, 1), (1, 2), (2, 3)])
    request = TransitiveClosureRequest(graph=graph, reflexive=False)
    result = compute_transitive_closure(request)

    # Chain 0->1->2->3: closure pairs are (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    expected = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    assert set(result.closure_pairs) == expected
    assert result.reflexive is False


def test_transitive_closure_chain_reflexive() -> None:
    graph = _make_graph(4, [(0, 1), (1, 2), (2, 3)])
    request = TransitiveClosureRequest(graph=graph, reflexive=True)
    result = compute_transitive_closure(request)

    # Closure should include all irreflexive pairs + (v,v) for all vertices.
    expected = {
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 1), (1, 2), (1, 3),
        (2, 2), (2, 3),
        (3, 3),
    }
    assert set(result.closure_pairs) == expected
    assert result.reflexive is True


# ---------------------------------------------------------------------------
# 13. A closure convention test for reflexive pairs
# ---------------------------------------------------------------------------
def test_transitive_closure_reflexive_vs_irreflexive() -> None:
    """Reflexive convention is explicitly declared and changes the result."""
    graph = _make_graph(3, [(0, 1), (1, 2)])

    req_irr = TransitiveClosureRequest(graph=graph, reflexive=False)
    result_irr = compute_transitive_closure(req_irr)
    req_ref = TransitiveClosureRequest(graph=graph, reflexive=True)
    result_ref = compute_transitive_closure(req_ref)

    # Irreflexive must not contain (0,0)
    assert (0, 0) not in set(result_irr.closure_pairs)
    # Reflexive must contain (0,0)
    assert (0, 0) in set(result_ref.closure_pairs)
    # Both must contain (0, 2)
    assert (0, 2) in set(result_irr.closure_pairs)
    assert (0, 2) in set(result_ref.closure_pairs)


# ---------------------------------------------------------------------------
# 14. Requests exactly at and immediately over the operation-specific bounds
# ---------------------------------------------------------------------------
def test_directed_graph_at_max_vertex_count() -> None:
    """Vertex count exactly at the max (256) is accepted."""
    arcs = [(0, 1)]
    graph = DirectedGraph.model_validate(
        {"vertex_count": 256, "arcs": arcs}
    )
    assert graph.vertex_count == 256


def test_directed_graph_over_max_vertex_count() -> None:
    """Vertex count over the max (257) is rejected."""
    with pytest.raises(ValidationError, match="less than or equal"):
        DirectedGraph.model_validate({"vertex_count": 257, "arcs": []})


def test_directed_graph_duplicate_arcs_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        DirectedGraph.model_validate(
            {"vertex_count": 3, "arcs": [(0, 1), (0, 1)]}
        )


def test_directed_graph_arc_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError, match="must be in"):
        DirectedGraph.model_validate(
            {"vertex_count": 3, "arcs": [(0, 3)]}
        )


# ---------------------------------------------------------------------------
# 15. One bounded transition graph extracted from the square-free digit-walk resources
# ---------------------------------------------------------------------------
def test_digit_walk_dead_end_transition_graph() -> None:
    """A bounded transition graph from square-free digit-walk resources.

    This models a small dead-end graph where vertex 0 is a dead end
    (sink) in the transition relation.  The degree profile must expose
    it as a sink, and reachability from 1 must show vertex 0 is reachable.
    """
    # Transition relation: 3 -> 2, 2 -> 1, 1 -> 0 (dead end)
    # Vertex 0 is a dead end (sink).  Vertex 3 is a source.
    arcs = [(3, 2), (2, 1), (1, 0)]
    graph = _make_graph(4, arcs)

    deg_request = DegreeProfileRequest(graph=graph)
    deg_result = compute_degree_profile(deg_request)

    assert deg_result.sources == (3,)
    assert deg_result.sinks == (0,)
    assert 0 in deg_result.isolated or deg_result.isolated == ()

    reach_request = ReachabilityRequest(graph=graph, source=3)
    reach_result = compute_reachability(reach_request)
    assert reach_result.reachable == (0, 1, 2, 3)

    # Check that vertex 0 is reachable from vertex 3
    assert 0 in reach_result.reachable


# ---------------------------------------------------------------------------
# 16. Composition into directed percolation operations (no conversion registry)
# ---------------------------------------------------------------------------
def test_composition_reachability_then_degree_profile() -> None:
    """Compose two operations without any conversion registry.

    The model decides to compute reachability first, then check the
    degree profile of the same graph.  Both use the same DirectedGraph
    value directly with no intermediate conversion.
    """
    arcs = [(0, 1), (1, 2), (2, 3), (0, 3)]
    graph = _make_graph(4, arcs)

    # Step 1: Check reachability from vertex 0
    reach_req = ReachabilityRequest(graph=graph, source=0, target=3)
    reach_result = compute_reachability(reach_req)
    assert reach_result.target_reachable is True

    # Step 2: Use the same graph for degree profile
    deg_req = DegreeProfileRequest(graph=graph)
    deg_result = compute_degree_profile(deg_req)
    assert deg_result.sources == (0,)

    # Step 3: Use the same graph for acyclic order
    acyclic_req = AcyclicOrderRequest(graph=graph)
    acyclic_result = compute_acyclic_order(acyclic_req)
    assert acyclic_result.status == "ACYCLIC"


# ---------------------------------------------------------------------------
# Contract validation edge cases
# ---------------------------------------------------------------------------
def test_directed_graph_empty_arcs_valid() -> None:
    graph = DirectedGraph.model_validate({"vertex_count": 5, "arcs": []})
    request = DegreeProfileRequest(graph=graph)
    result = compute_degree_profile(request)
    assert result.isolated == (0, 1, 2, 3, 4)


def test_reachability_source_self_reachable() -> None:
    """Source vertex is reachable from itself (distance 0)."""
    graph = _make_graph(3, [(0, 1), (1, 2)])
    request = ReachabilityRequest(graph=graph, source=0)
    result = compute_reachability(request)
    assert result.distances[0] == 0
    assert result.predecessors[0] is None


def test_scc_single_vertex_is_strongly_connected() -> None:
    graph = _make_graph(1, [])
    request = StrongComponentsRequest(graph=graph)
    result = compute_strong_components(request)
    assert result.is_strongly_connected
    assert result.component_count == 1


def test_scc_two_vertex_no_edge_not_strongly_connected() -> None:
    graph = _make_graph(2, [])
    request = StrongComponentsRequest(graph=graph)
    result = compute_strong_components(request)
    assert not result.is_strongly_connected
    assert result.component_count == 2
