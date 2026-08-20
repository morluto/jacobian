"""Tests for network optimization operations."""

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.flow._models import (
    CostedFlowEdge,
    CostedFlowGraph,
    MinCostFlowRequest,
)
from jacobian.math.graphs.flow._operations import (
    compute_min_cost_flow,
)
from jacobian.math.graphs.flow._tools import TOOLS


def test_catalog_contains_only_audited_operations() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "graph.flow.maximum.compute",
        "graph.cut.minimum_st.compute",
        "graph.menger.edge_disjoint.compute",
        "network.min_cost_flow.compute",
    }


def _make_graph(edges_data):
    edges = tuple(
        CostedFlowEdge(
            source=s,
            target=t,
            capacity=CanonicalRational(num=c, den="1"),
            cost=CanonicalRational(num=co, den="1"),
        )
        for s, t, c, co in edges_data
    )
    return CostedFlowGraph(
        vertex_count=max(max(s, t) for s, t, _, _ in edges_data) + 1, edges=edges
    )


def test_min_cost_flow_basic() -> None:
    graph = CostedFlowGraph(
        vertex_count=3,
        edges=(
            CostedFlowEdge(
                source=0,
                target=1,
                capacity=CanonicalRational(num="5", den="1"),
                cost=CanonicalRational(num="1", den="1"),
            ),
            CostedFlowEdge(
                source=1,
                target=2,
                capacity=CanonicalRational(num="5", den="1"),
                cost=CanonicalRational(num="2", den="1"),
            ),
            CostedFlowEdge(
                source=0,
                target=2,
                capacity=CanonicalRational(num="5", den="1"),
                cost=CanonicalRational(num="4", den="1"),
            ),
        ),
    )
    request = MinCostFlowRequest(graph=graph, demands=(-2, 0, 2))
    result = compute_min_cost_flow(request)
    assert result.feasible is True
    assert result.total_cost.as_fraction() == 6


def test_min_cost_flow_exact_rationals() -> None:
    """Verify exact rational arithmetic with fractional capacities/costs."""
    graph = CostedFlowGraph(
        vertex_count=3,
        edges=(
            CostedFlowEdge(
                source=0,
                target=1,
                capacity=CanonicalRational(num="5", den="2"),
                cost=CanonicalRational(num="1", den="3"),
            ),
            CostedFlowEdge(
                source=1,
                target=2,
                capacity=CanonicalRational(num="5", den="2"),
                cost=CanonicalRational(num="2", den="3"),
            ),
            CostedFlowEdge(
                source=0,
                target=2,
                capacity=CanonicalRational(num="5", den="2"),
                cost=CanonicalRational(num="4", den="3"),
            ),
        ),
    )
    request = MinCostFlowRequest(graph=graph, demands=(-2, 0, 2))
    result = compute_min_cost_flow(request)
    assert result.feasible is True
    # Flow should go through the cheaper path: 0->1->2 at cost 1/3 + 2/3 = 1 per unit
    # 2 units at cost 1 = 2
    assert result.total_cost.as_fraction() == Fraction(2)


def test_min_cost_flow_conservation() -> None:
    """Verify flow conservation: sum of flow in - sum of flow out = demand."""
    graph = CostedFlowGraph(
        vertex_count=3,
        edges=(
            CostedFlowEdge(
                source=0,
                target=1,
                capacity=CanonicalRational(num="5", den="1"),
                cost=CanonicalRational(num="1", den="1"),
            ),
            CostedFlowEdge(
                source=1,
                target=2,
                capacity=CanonicalRational(num="5", den="1"),
                cost=CanonicalRational(num="2", den="1"),
            ),
            CostedFlowEdge(
                source=0,
                target=2,
                capacity=CanonicalRational(num="5", den="1"),
                cost=CanonicalRational(num="4", den="1"),
            ),
        ),
    )
    request = MinCostFlowRequest(graph=graph, demands=(-2, 0, 2))
    result = compute_min_cost_flow(request)
    assert result.feasible is True

    # Verify conservation at each vertex
    vertex_count = request.graph.vertex_count
    balance = [Fraction(0)] * vertex_count
    for fe in result.flow_edges:
        balance[fe.source] -= fe.flow.as_fraction()
        balance[fe.target] += fe.flow.as_fraction()
    for v, d in enumerate(request.demands):
        assert balance[v] == d, (
            f"conservation violated at vertex {v}: {balance[v]} != {d}"
        )

    # Verify capacity constraints
    edge_map = {}
    for e in request.graph.edges:
        edge_map[(e.source, e.target)] = e.capacity.as_fraction()
    for fe in result.flow_edges:
        cap = edge_map.get((fe.source, fe.target), Fraction(0))
        assert 0 <= fe.flow.as_fraction() <= cap, (
            f"capacity violated on edge {fe.source}->{fe.target}"
        )

    # Verify total cost
    cost_map = {}
    for e in request.graph.edges:
        cost_map[(e.source, e.target)] = e.cost.as_fraction()
    total = sum(
        cost_map[(fe.source, fe.target)] * fe.flow.as_fraction()
        for fe in result.flow_edges
    )
    assert total == result.total_cost.as_fraction(), (
        "reported total cost does not match recomputed cost"
    )


def test_min_cost_flow_infeasible() -> None:
    graph = CostedFlowGraph(
        vertex_count=3,
        edges=(
            CostedFlowEdge(
                source=0,
                target=1,
                capacity=CanonicalRational(num="1", den="1"),
                cost=CanonicalRational(num="1", den="1"),
            ),
            CostedFlowEdge(
                source=1,
                target=2,
                capacity=CanonicalRational(num="1", den="1"),
                cost=CanonicalRational(num="2", den="1"),
            ),
        ),
    )
    request = MinCostFlowRequest(graph=graph, demands=(-10, 0, 10))
    result = compute_min_cost_flow(request)
    assert result.feasible is False
