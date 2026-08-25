"""Tests for network optimization operations."""

import copy
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.flow._models import (
    CostedFlowEdge,
    CostedFlowGraph,
    MinCostFlowRequest,
    MinCostFlowResult,
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
    assert result.flow_edges == ()
    assert result.total_cost.as_fraction() == 0


def test_min_cost_flow_fractional_capacity_below_demand_is_infeasible() -> None:
    """Issue #2292: demand 1 cannot cross a capacity-1/2 edge.

    The scaling bug admitted this request and returned flow 1 across the
    half-unit edge; the source problem is infeasible.
    """
    graph = CostedFlowGraph(
        vertex_count=2,
        edges=(
            CostedFlowEdge(
                source=0,
                target=1,
                capacity=CanonicalRational(num="1", den="2"),
                cost=CanonicalRational(num="1", den="3"),
            ),
        ),
    )
    result = compute_min_cost_flow(MinCostFlowRequest(graph=graph, demands=(-1, 1)))
    assert result.feasible is False
    assert result.flow_edges == ()
    assert result.total_cost.as_fraction() == 0


def test_min_cost_flow_rational_flow_and_objective_use_distinct_scales() -> None:
    """A feasible fractional flow is divided back through the flow scale."""
    graph = CostedFlowGraph(
        vertex_count=2,
        edges=(
            CostedFlowEdge(
                source=0,
                target=1,
                capacity=CanonicalRational(num="3", den="2"),
                cost=CanonicalRational(num="2", den="3"),
            ),
        ),
    )
    result = compute_min_cost_flow(MinCostFlowRequest(graph=graph, demands=(-1, 1)))
    assert result.feasible is True
    assert [(e.source, e.target, e.flow.as_fraction()) for e in result.flow_edges] == [
        (0, 1, Fraction(1, 1))
    ]
    assert result.total_cost.as_fraction() == Fraction(2, 3)


def test_min_cost_flow_mixed_capacity_and_cost_denominators() -> None:
    """Distinct flow and cost scales compose exactly across edges."""
    graph = CostedFlowGraph(
        vertex_count=3,
        edges=(
            CostedFlowEdge(
                source=0,
                target=1,
                capacity=CanonicalRational(num="3", den="2"),
                cost=CanonicalRational(num="2", den="3"),
            ),
            CostedFlowEdge(
                source=1,
                target=2,
                capacity=CanonicalRational(num="5", den="3"),
                cost=CanonicalRational(num="4", den="5"),
            ),
        ),
    )
    result = compute_min_cost_flow(MinCostFlowRequest(graph=graph, demands=(-1, 0, 1)))
    assert result.feasible is True
    assert result.total_cost.as_fraction() == Fraction(2, 3) + Fraction(4, 5)
    # The retained source network replays balances, capacities, and objective.
    balance = [Fraction(0)] * 3
    for fe in result.flow_edges:
        balance[fe.source] -= fe.flow.as_fraction()
        balance[fe.target] += fe.flow.as_fraction()
    assert balance == [Fraction(-1), Fraction(0), Fraction(1)]
    assert result.graph == graph
    assert result.demands == (-1, 0, 1)


def _two_path_graph() -> CostedFlowGraph:
    return CostedFlowGraph(
        vertex_count=3,
        edges=(
            CostedFlowEdge(
                source=0,
                target=1,
                capacity=CanonicalRational(num="5", den="2"),
                cost=CanonicalRational(num="1", den="3"),
            ),
            CostedFlowEdge(
                source=0,
                target=2,
                capacity=CanonicalRational(num="5", den="1"),
                cost=CanonicalRational(num="4", den="1"),
            ),
            CostedFlowEdge(
                source=2,
                target=1,
                capacity=CanonicalRational(num="5", den="1"),
                cost=CanonicalRational(num="1", den="1"),
            ),
        ),
    )


def test_min_cost_flow_serialized_result_replays_against_source() -> None:
    """Serialization round-trips and mutations of any field are rejected."""
    request = MinCostFlowRequest(graph=_two_path_graph(), demands=(-2, 2, 0))
    result = compute_min_cost_flow(request)
    assert result.feasible is True
    dumped = result.model_dump()
    assert MinCostFlowResult.model_validate(dumped) == result

    shrunk = copy.deepcopy(dumped)
    shrunk["graph"]["edges"][0]["capacity"] = {"num": "1", "den": "1"}
    with pytest.raises(ValidationError):
        MinCostFlowResult.model_validate(shrunk)

    undeclared = copy.deepcopy(dumped)
    undeclared["flow_edges"] = (
        *undeclared["flow_edges"],
        {"source": 1, "target": 0, "flow": {"num": "1", "den": "1"}},
    )
    with pytest.raises(ValidationError):
        MinCostFlowResult.model_validate(undeclared)

    rebalanced = copy.deepcopy(dumped)
    rebalanced["demands"] = [1, -2, 1]
    with pytest.raises(ValidationError):
        MinCostFlowResult.model_validate(rebalanced)

    recompensed = copy.deepcopy(dumped)
    recompensed["total_cost"] = {"num": "9999", "den": "1"}
    with pytest.raises(ValidationError):
        MinCostFlowResult.model_validate(recompensed)


def test_min_cost_flow_infeasible_result_carries_no_claim() -> None:
    with pytest.raises(ValidationError):
        MinCostFlowResult(
            graph=_two_path_graph(),
            demands=(-2, 2, 0),
            total_cost=CanonicalRational(num="7", den="1"),
            feasible=False,
        )


def test_min_cost_flow_derived_scale_admission_fails_closed() -> None:
    """Oversized derived LCMs are rejected before backend construction."""
    primes = []
    candidate = 2
    while len(primes) < 500:
        if all(candidate % p for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    pairs = [(i, j) for i in range(64) for j in range(64) if i != j][: len(primes)]
    edges = tuple(
        CostedFlowEdge(
            source=source,
            target=target,
            capacity=CanonicalRational(num="1", den=str(prime**3)),
            cost=CanonicalRational(num="1", den="1"),
        )
        for (source, target), prime in zip(pairs, primes, strict=True)
    )
    graph = CostedFlowGraph(vertex_count=64, edges=edges)
    with pytest.raises(ValidationError):
        MinCostFlowRequest(graph=graph, demands=tuple([0] * 63 + [0]))
