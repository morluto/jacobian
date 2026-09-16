"""Contract tests for exact multicommodity-flow witness checking."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.dispatch import parse_operation_input
from jacobian.math.graphs.flows._models import CapacitatedEdge, FlowGraph
from jacobian.math.graphs.flows.multicommodity._models import (
    CommodityDemand,
    CommodityEdgeFlow,
    MulticommodityFlow,
    MulticommodityFlowWitnessCheckRequest,
    MulticommodityFlowWitnessCheckResult,
)
from jacobian.math.graphs.flows.multicommodity._tools import (
    TOOLS,
    _run_multicommodity_flow_witness_check,
)
from jacobian.math.graphs.flows.multicommodity.operations import (
    check_multicommodity_flow_witness,
)


def q(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(numerator, denominator))


def shared_bottleneck() -> MulticommodityFlow:
    return MulticommodityFlow(
        network=FlowGraph(
            vertex_count=4,
            edges=(
                CapacitatedEdge(source=0, target=2, capacity=q(2)),
                CapacitatedEdge(source=1, target=2, capacity=q(2)),
                CapacitatedEdge(source=2, target=3, capacity=q(3)),
            ),
        ),
        commodities=(
            CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(1)),
            CommodityDemand(commodity_id="b", source=1, sink=3, demand=q(2)),
        ),
        entries=(
            CommodityEdgeFlow(commodity_id="a", source=0, target=2, amount=q(1)),
            CommodityEdgeFlow(commodity_id="a", source=2, target=3, amount=q(1)),
            CommodityEdgeFlow(commodity_id="b", source=1, target=2, amount=q(2)),
            CommodityEdgeFlow(commodity_id="b", source=2, target=3, amount=q(2)),
        ),
    )


def test_catalog_contains_the_witness_checker() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "network.multicommodity_flow.profile.compute",
        "network.multicommodity_flow.witness.check",
    }


def test_feasible_witness_known_answer() -> None:
    result = check_multicommodity_flow_witness(shared_bottleneck())
    assert result.status == "FEASIBLE"
    assert result.all_demands_routed is True
    assert result.capacity_feasible is True
    assert result.nonnegative is True
    assert result.congestion == q(1)
    assert result.violating_commodities == ()
    assert result.violating_vertices == ()
    assert result.violating_edges == ()
    loads = {(row.source, row.target): row.load for row in result.edge_profiles}
    assert loads == {(0, 2): q(1), (1, 2): q(2), (2, 3): q(3)}


def test_conservation_violation_is_reported_exactly() -> None:
    flow = MulticommodityFlow(
        network=shared_bottleneck().network,
        commodities=shared_bottleneck().commodities,
        entries=(
            CommodityEdgeFlow(commodity_id="a", source=0, target=2, amount=q(1)),
            CommodityEdgeFlow(commodity_id="a", source=2, target=3, amount=q(1)),
            CommodityEdgeFlow(commodity_id="b", source=1, target=2, amount=q(1)),
            CommodityEdgeFlow(commodity_id="b", source=2, target=3, amount=q(2)),
        ),
    )
    result = check_multicommodity_flow_witness(flow)
    assert result.status == "INFEASIBLE"
    assert result.all_demands_routed is False
    assert result.violating_commodities == ("b",)
    assert {(row.vertex, row.divergence) for row in result.violating_vertices} == {
        (1, q(1)),
        (2, q(1)),
    }
    assert result.violating_edges == ()


def test_capacity_violation_is_reported_exactly() -> None:
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=2,
            edges=(CapacitatedEdge(source=0, target=1, capacity=q(1)),),
        ),
        commodities=(
            CommodityDemand(commodity_id="a", source=0, sink=1, demand=q(1)),
            CommodityDemand(commodity_id="b", source=0, sink=1, demand=q(1)),
        ),
        entries=(
            CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=q(1)),
            CommodityEdgeFlow(commodity_id="b", source=0, target=1, amount=q(1)),
        ),
    )
    result = check_multicommodity_flow_witness(flow)
    assert result.status == "INFEASIBLE"
    assert result.all_demands_routed is True
    assert result.capacity_feasible is False
    assert len(result.violating_edges) == 1
    violation = result.violating_edges[0]
    assert (violation.source, violation.target) == (0, 1)
    assert violation.load == q(2)
    assert violation.capacity == q(1)


def test_zero_capacity_edge_with_flow_is_infeasible_and_without_is_feasible() -> None:
    network = FlowGraph(
        vertex_count=2,
        edges=(CapacitatedEdge(source=0, target=1, capacity=q(0)),),
    )
    commodities = (CommodityDemand(commodity_id="a", source=0, sink=1, demand=q(1)),)
    loaded = MulticommodityFlow(
        network=network,
        commodities=commodities,
        entries=(CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=q(1)),),
    )
    result = check_multicommodity_flow_witness(loaded)
    assert result.status == "INFEASIBLE"
    assert result.congestion is None
    assert len(result.violating_edges) == 1
    empty = MulticommodityFlow(network=network, commodities=commodities, entries=())
    empty_result = check_multicommodity_flow_witness(empty)
    assert empty_result.status == "INFEASIBLE"
    assert empty_result.violating_commodities == ("a",)


def test_fractional_splitting_is_feasible_exactly() -> None:
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=4,
            edges=(
                CapacitatedEdge(source=0, target=1, capacity=q(1)),
                CapacitatedEdge(source=0, target=2, capacity=q(1)),
                CapacitatedEdge(source=1, target=3, capacity=q(1)),
                CapacitatedEdge(source=2, target=3, capacity=q(1)),
            ),
        ),
        commodities=(CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(1)),),
        entries=(
            CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=q(1, 2)),
            CommodityEdgeFlow(commodity_id="a", source=0, target=2, amount=q(1, 2)),
            CommodityEdgeFlow(commodity_id="a", source=1, target=3, amount=q(1, 2)),
            CommodityEdgeFlow(commodity_id="a", source=2, target=3, amount=q(1, 2)),
        ),
    )
    result = check_multicommodity_flow_witness(flow)
    assert result.status == "FEASIBLE"
    assert result.congestion == q(1, 2)


def test_serialization_round_trip() -> None:
    result = check_multicommodity_flow_witness(shared_bottleneck())
    replayed = MulticommodityFlowWitnessCheckResult.model_validate_json(
        result.model_dump_json()
    )
    assert replayed == result
    infeasible = check_multicommodity_flow_witness(
        MulticommodityFlow(
            network=shared_bottleneck().network,
            commodities=shared_bottleneck().commodities,
            entries=(
                CommodityEdgeFlow(commodity_id="a", source=0, target=2, amount=q(1)),
                CommodityEdgeFlow(commodity_id="a", source=2, target=3, amount=q(1)),
                CommodityEdgeFlow(commodity_id="b", source=1, target=2, amount=q(1)),
                CommodityEdgeFlow(commodity_id="b", source=2, target=3, amount=q(2)),
            ),
        )
    )
    replayed_infeasible = MulticommodityFlowWitnessCheckResult.model_validate_json(
        infeasible.model_dump_json()
    )
    assert replayed_infeasible == infeasible


def test_catalog_and_native_paths_agree() -> None:
    example = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "network.multicommodity_flow.witness.check"
    ).examples[0]
    request = parse_operation_input(
        MulticommodityFlowWitnessCheckRequest, example.input
    )
    catalog_result = _run_multicommodity_flow_witness_check(request)
    native_result = check_multicommodity_flow_witness(shared_bottleneck())
    assert catalog_result == native_result


def test_undeclared_edge_entry_is_structurally_rejected() -> None:
    with pytest.raises(ValidationError) as rejected:
        MulticommodityFlow(
            network=FlowGraph(
                vertex_count=3,
                edges=(CapacitatedEdge(source=0, target=1, capacity=q(1)),),
            ),
            commodities=(
                CommodityDemand(commodity_id="a", source=0, sink=2, demand=q(1)),
            ),
            entries=(
                CommodityEdgeFlow(commodity_id="a", source=1, target=2, amount=q(1)),
            ),
        )
    assert rejected.value.errors()[0]["type"].startswith("graph.")


def test_undeclared_commodity_entry_is_structurally_rejected() -> None:
    with pytest.raises(ValidationError) as rejected:
        MulticommodityFlow(
            network=FlowGraph(
                vertex_count=2,
                edges=(CapacitatedEdge(source=0, target=1, capacity=q(1)),),
            ),
            commodities=(
                CommodityDemand(commodity_id="a", source=0, sink=1, demand=q(1)),
            ),
            entries=(
                CommodityEdgeFlow(commodity_id="b", source=0, target=1, amount=q(1)),
            ),
        )
    assert rejected.value.errors()[0]["type"].startswith("graph.")
