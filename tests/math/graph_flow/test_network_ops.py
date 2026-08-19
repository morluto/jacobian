"""Tests for network optimization operations."""

from jacobian.math.graphs.flow._models import (
    CirculationRequest,
    CostedFlowEdge,
    CostedFlowGraph,
    MinCostFlowRequest,
)
from jacobian.math.graphs.flow._operations import (
    compute_circulation,
    compute_min_cost_flow,
)


def test_min_cost_flow_basic() -> None:
    graph = CostedFlowGraph(
        vertex_count=3,
        edges=(
            CostedFlowEdge(
                source=0,
                target=1,
                capacity=__import__(
                    "jacobian._exact", fromlist=["CanonicalRational"]
                ).CanonicalRational(num="5", den="1"),
                cost=__import__(
                    "jacobian._exact", fromlist=["CanonicalRational"]
                ).CanonicalRational(num="1", den="1"),
            ),
            CostedFlowEdge(
                source=1,
                target=2,
                capacity=__import__(
                    "jacobian._exact", fromlist=["CanonicalRational"]
                ).CanonicalRational(num="5", den="1"),
                cost=__import__(
                    "jacobian._exact", fromlist=["CanonicalRational"]
                ).CanonicalRational(num="2", den="1"),
            ),
            CostedFlowEdge(
                source=0,
                target=2,
                capacity=__import__(
                    "jacobian._exact", fromlist=["CanonicalRational"]
                ).CanonicalRational(num="5", den="1"),
                cost=__import__(
                    "jacobian._exact", fromlist=["CanonicalRational"]
                ).CanonicalRational(num="4", den="1"),
            ),
        ),
    )
    request = MinCostFlowRequest(graph=graph, demands=(-2, 0, 2))
    result = compute_min_cost_flow(request)
    assert result.feasible is True
    assert result.total_cost.as_fraction() == 6


def test_circulation_feasible() -> None:
    from jacobian._exact import CanonicalRational

    graph = CostedFlowGraph(
        vertex_count=2,
        edges=(
            CostedFlowEdge(
                source=0,
                target=1,
                capacity=CanonicalRational(num="1", den="1"),
                cost=CanonicalRational(num="0", den="1"),
            ),
            CostedFlowEdge(
                source=1,
                target=0,
                capacity=CanonicalRational(num="1", den="1"),
                cost=CanonicalRational(num="0", den="1"),
            ),
        ),
    )
    request = CirculationRequest(graph=graph)
    result = compute_circulation(request)
    assert result.feasible is True
