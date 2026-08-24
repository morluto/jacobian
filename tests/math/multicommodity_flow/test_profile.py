"""Contract tests for exact bounded multicommodity-flow profiles."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.flow._models import CapacitatedEdge, FlowGraph
from jacobian.math.graphs.multicommodity_flow._models import (
    CommodityDemand,
    CommodityEdgeFlow,
    MulticommodityFlow,
    MulticommodityFlowProfileRequest,
    MulticommodityFlowProfileResult,
)
from jacobian.math.graphs.multicommodity_flow._operations import (
    compute_multicommodity_flow_profile,
)
from jacobian.math.graphs.multicommodity_flow._tools import TOOLS


def q(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(numerator, denominator))


def shared_bottleneck_flow() -> MulticommodityFlow:
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


def test_catalog_contains_the_audited_multicommodity_profile() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "network.multicommodity_flow.profile.compute"
    }


def test_exact_shared_bottleneck_profile_replays_its_source() -> None:
    result = compute_multicommodity_flow_profile(
        MulticommodityFlowProfileRequest(flow=shared_bottleneck_flow())
    )

    assert result.all_demands_routed is True
    assert result.capacity_feasible is True
    assert result.congestion == q(1)
    assert [
        (row.source, row.target, row.load.as_fraction(), row.slack.as_fraction())
        for row in result.edge_profiles
    ] == [
        (0, 2, Fraction(1), Fraction(1)),
        (1, 2, Fraction(2), Fraction(0)),
        (2, 3, Fraction(3), Fraction(0)),
    ]
    assert [
        (row.commodity_id, row.vertex, row.divergence.as_fraction())
        for row in result.divergences
    ] == [
        ("a", 0, Fraction(1)),
        ("a", 1, Fraction(0)),
        ("a", 2, Fraction(0)),
        ("a", 3, Fraction(-1)),
        ("b", 0, Fraction(0)),
        ("b", 1, Fraction(2)),
        ("b", 2, Fraction(0)),
        ("b", 3, Fraction(-2)),
    ]
    # Producer and replay both scan the four sparse entries, four-by-two dense
    # divergence cells, and three edges; the ledger is exact rather than a cap.
    assert result.work.sparse_entries == 4
    assert result.work.commodity_vertex_cells == 8
    assert result.work.edge_cells == 3
    assert result.work.rational_additions_per_pass == 15
    assert result.work.rational_negations_per_pass == 2
    assert result.work.rational_divisions_per_pass == 3
    assert result.work.exact_comparisons_per_pass == 17
    assert result.work.logical_steps_per_call == 74


def test_profile_distinguishes_capacity_and_conservation_failures() -> None:
    capacity_only = shared_bottleneck_flow().model_copy(
        update={
            "network": FlowGraph(
                vertex_count=4,
                edges=(
                    CapacitatedEdge(source=0, target=2, capacity=q(2)),
                    CapacitatedEdge(source=1, target=2, capacity=q(2)),
                    CapacitatedEdge(source=2, target=3, capacity=q(2)),
                ),
            )
        }
    )
    capacity_result = compute_multicommodity_flow_profile(
        MulticommodityFlowProfileRequest(flow=capacity_only)
    )
    assert capacity_result.all_demands_routed is True
    assert capacity_result.capacity_feasible is False
    assert capacity_result.congestion == q(3, 2)

    conservation_only = shared_bottleneck_flow().model_copy(
        update={
            "entries": (
                CommodityEdgeFlow(commodity_id="a", source=0, target=2, amount=q(1)),
                CommodityEdgeFlow(commodity_id="a", source=2, target=3, amount=q(1)),
                CommodityEdgeFlow(commodity_id="b", source=1, target=2, amount=q(2)),
            )
        }
    )
    conservation_result = compute_multicommodity_flow_profile(
        MulticommodityFlowProfileRequest(flow=conservation_only)
    )
    assert conservation_result.all_demands_routed is False
    assert conservation_result.capacity_feasible is True


def test_zero_capacity_positive_load_has_no_finite_congestion() -> None:
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=2,
            edges=(CapacitatedEdge(source=0, target=1, capacity=q(0)),),
        ),
        commodities=(CommodityDemand(commodity_id="a", source=0, sink=1, demand=q(1)),),
        entries=(CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=q(1)),),
    )
    result = compute_multicommodity_flow_profile(
        MulticommodityFlowProfileRequest(flow=flow)
    )
    assert result.all_demands_routed is True
    assert result.capacity_feasible is False
    assert result.congestion is None
    # A zero-capacity edge still compares load to capacity and capacity to zero,
    # then checks whether its load is positive; no ratio division is attempted.
    assert result.work.rational_divisions_per_pass == 0
    assert result.work.exact_comparisons_per_pass == 5
    assert result.work.logical_steps_per_call == 20


def test_fractional_split_flow_uses_exact_rational_loads_and_congestion() -> None:
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=4,
            edges=(
                CapacitatedEdge(source=0, target=1, capacity=q(1, 2)),
                CapacitatedEdge(source=0, target=2, capacity=q(1, 2)),
                CapacitatedEdge(source=1, target=3, capacity=q(1, 2)),
                CapacitatedEdge(source=2, target=3, capacity=q(1, 2)),
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
    result = compute_multicommodity_flow_profile(
        MulticommodityFlowProfileRequest(flow=flow)
    )
    assert result.all_demands_routed is True
    assert result.capacity_feasible is True
    assert result.congestion == q(1)
    assert all(row.slack == q(0) for row in result.edge_profiles)


def test_sparse_tensor_rejects_zero_duplicate_unknown_and_unsorted_cells() -> None:
    payload = shared_bottleneck_flow().model_dump(mode="json")

    zero = deepcopy(payload)
    zero["entries"][0]["amount"] = {"num": "0", "den": "1"}
    with pytest.raises(ValidationError, match="strictly positive"):
        MulticommodityFlow.model_validate(zero)

    duplicate = deepcopy(payload)
    duplicate["entries"].append(deepcopy(duplicate["entries"][0]))
    with pytest.raises(ValidationError, match=r"sorted|once"):
        MulticommodityFlow.model_validate(duplicate)

    undeclared = deepcopy(payload)
    undeclared["entries"][1]["commodity_id"] = "aa"
    with pytest.raises(ValidationError, match="undeclared commodity"):
        MulticommodityFlow.model_validate(undeclared)

    unsorted = deepcopy(payload)
    unsorted["entries"] = list(reversed(unsorted["entries"]))
    with pytest.raises(ValidationError, match="sorted"):
        MulticommodityFlow.model_validate(unsorted)


def test_tighter_multicommodity_envelope_rejects_vertex_and_rational_overflow() -> None:
    payload = shared_bottleneck_flow().model_dump(mode="json")

    too_many_vertices = deepcopy(payload)
    too_many_vertices["network"]["vertex_count"] = 33
    with pytest.raises(ValidationError, match="at most 32 vertices"):
        MulticommodityFlow.model_validate(too_many_vertices)

    too_many_digits = deepcopy(payload)
    too_many_digits["network"]["edges"][0]["capacity"] = {
        "num": str(10**32),
        "den": "1",
    }
    with pytest.raises(ValidationError, match="32-digit"):
        MulticommodityFlow.model_validate(too_many_digits)


def test_result_replay_rejects_forged_source_and_derived_ledger_fields() -> None:
    result = compute_multicommodity_flow_profile(
        MulticommodityFlowProfileRequest(flow=shared_bottleneck_flow())
    )
    payload = result.model_dump(mode="json")

    forged_load = deepcopy(payload)
    forged_load["edge_profiles"][2]["load"] = {"num": "2", "den": "1"}
    with pytest.raises(ValidationError, match="exact multicommodity-flow profile"):
        MulticommodityFlowProfileResult.model_validate(forged_load)

    forged_source = deepcopy(payload)
    forged_source["flow"]["commodities"][1]["demand"] = {"num": "1", "den": "1"}
    with pytest.raises(ValidationError, match="exact multicommodity-flow profile"):
        MulticommodityFlowProfileResult.model_validate(forged_source)

    forged_work = deepcopy(payload)
    forged_work["work"]["logical_steps_per_call"] = 1
    with pytest.raises(ValidationError, match="exact multicommodity-flow profile"):
        MulticommodityFlowProfileResult.model_validate(forged_work)
