"""Contract tests for exact bounded multicommodity-flow profiles."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from fractions import Fraction

import pytest
from pydantic import ValidationError
from tests.fixtures.accounting import assert_charged_work_parity
from tests.math.graphs.flows.multicommodity._support import (
    multicommodity_validation_error,
)

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.math.graphs.flows._models import CapacitatedEdge, FlowGraph
from jacobian.math.graphs.flows.multicommodity._models import (
    MAX_COMMODITY_VERTEX_CELLS,
    MAX_PROFILE_COMPARISONS_PER_PASS,
    MAX_PROFILE_LOGICAL_STEPS,
    CommodityDemand,
    CommodityEdgeFlow,
    MulticommodityFlow,
    MulticommodityFlowProfileRequest,
    MulticommodityFlowProfileResult,
    derived_profile_digit_budget,
    measured_profile_components,
)
from jacobian.math.graphs.flows.multicommodity._tools import (
    TOOLS,
    _run_multicommodity_flow_profile,
)
from jacobian.math.graphs.flows.multicommodity.operations import (
    compute_multicommodity_flow_profile,
)


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


def test_native_api_accepts_the_canonical_flow_value_directly() -> None:
    flow = shared_bottleneck_flow()

    native = compute_multicommodity_flow_profile(flow)
    via_request = _run_multicommodity_flow_profile(
        MulticommodityFlowProfileRequest(flow=flow)
    )

    assert native == via_request
    assert native.flow == flow
    assert native.congestion == q(1)


def test_accepted_calls_charge_every_executed_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Request parsing performs the operation's component scan once, admits
    # work and result envelope from it, then hands it to the producer. A
    # native call runs that same one charged scan at its execution boundary.
    from jacobian.math.graphs.flows.multicommodity import _models

    scans = {"count": 0}
    original_models_scan = _models._component_sums_with_folds

    def models_spy(flow: MulticommodityFlow) -> object:
        scans["count"] += 1
        return original_models_scan(flow)

    monkeypatch.setattr(_models, "_component_sums_with_folds", models_spy)

    via_request = _run_multicommodity_flow_profile(
        MulticommodityFlowProfileRequest(flow=shared_bottleneck_flow())
    )
    assert_charged_work_parity(
        charged={"component_scan": 1}, executed={"component_scan": scans["count"]}
    )
    assert scans == {"count": 1}

    scans.update(count=0)
    native = compute_multicommodity_flow_profile(shared_bottleneck_flow())
    assert_charged_work_parity(
        charged={"component_scan": 1}, executed={"component_scan": scans["count"]}
    )
    assert scans == {"count": 1}

    assert native == via_request


def test_exact_shared_bottleneck_profile_has_its_expected_values() -> None:
    result = compute_multicommodity_flow_profile(shared_bottleneck_flow())

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
    # The producer scans the four sparse entries, four-by-two dense divergence
    # cells, and three edges; the ledger is exact rather than a cap.
    # Every entry shares denominator 1, so the bucket fold performs one
    # fraction addition per nonempty bucket: six touched divergence cells and
    # three edges add nine folds to the twelve numerator additions, plus the
    # three slack subtractions. The shared derivative scan classifies each
    # edge once and compares each positive-capacity ratio against its running
    # maximum once; the kernel loop adds one feasibility test per edge.
    assert result.work.sparse_entries == 4
    assert result.work.commodity_vertex_cells == 8
    assert result.work.edge_cells == 3
    assert result.work.rational_additions_per_pass == 3 * 4 + (6 + 3) + 3
    assert result.work.rational_negations_per_pass == 2
    assert result.work.rational_divisions_per_pass == 3
    assert result.work.exact_comparisons_per_pass == 8 + 3 * 3
    assert result.work.logical_steps_per_call == 24 + 2 + 3 + 17


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
    capacity_result = compute_multicommodity_flow_profile(capacity_only)
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
    conservation_result = compute_multicommodity_flow_profile(conservation_only)
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
    result = compute_multicommodity_flow_profile(flow)
    assert result.all_demands_routed is True
    assert result.capacity_feasible is False
    assert result.congestion is None
    # The shared derivative scan classifies the zero-capacity edge and checks
    # its load; because that load is positive no ratio is ever divided, and
    # the kernel loop adds only its load-vs-capacity feasibility test. The
    # lone entry folds one denominator into each of its two divergence cells
    # and one edge bucket: three fold additions beyond 3*1 + 1.
    assert result.work.rational_divisions_per_pass == 0
    assert result.work.exact_comparisons_per_pass == 2 + 1 + 2
    assert result.work.logical_steps_per_call == (3 + 3 + 1) + 1 + 0 + 5


def test_early_divergence_mismatch_still_executes_every_counted_comparison() -> None:
    # Commodity "a" already mismatches at the very first divergence cell, yet
    # the demand scan must compare all eight commodity/vertex cells rather
    # than short-circuiting while still charging them.
    flow = shared_bottleneck_flow().model_copy(
        update={
            "commodities": (
                CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(2)),
                CommodityDemand(commodity_id="b", source=1, sink=3, demand=q(2)),
            )
        }
    )
    result = compute_multicommodity_flow_profile(flow)
    assert result.all_demands_routed is False
    assert result.capacity_feasible is True
    assert result.work.commodity_vertex_cells == 8
    assert result.work.edge_cells == 3
    assert result.work.exact_comparisons_per_pass == 8 + 3 * 3
    assert result.work.logical_steps_per_call == 24 + 2 + 3 + 17


def test_mixed_zero_and_positive_capacities_execute_every_counted_comparison() -> None:
    def mixed_flow(zero_capacity_edge_first: bool) -> MulticommodityFlow:
        capacities = ((q(0), q(1), q(1)), (q(1), q(1), q(0)))[zero_capacity_edge_first]
        return MulticommodityFlow(
            network=FlowGraph(
                vertex_count=4,
                edges=(
                    CapacitatedEdge(source=0, target=1, capacity=capacities[0]),
                    CapacitatedEdge(source=1, target=2, capacity=capacities[1]),
                    CapacitatedEdge(source=2, target=3, capacity=capacities[2]),
                ),
            ),
            commodities=(
                CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(1)),
            ),
            entries=(
                CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=q(1)),
                CommodityEdgeFlow(commodity_id="a", source=1, target=2, amount=q(1)),
                CommodityEdgeFlow(commodity_id="a", source=2, target=3, amount=q(1)),
            ),
        )

    for zero_capacity_edge_first in (True, False):
        result = compute_multicommodity_flow_profile(
            mixed_flow(zero_capacity_edge_first)
        )
        assert result.all_demands_routed is True
        assert result.capacity_feasible is False
        # The loaded zero-capacity edge makes the congestion value null, so
        # the shared derivative scan divides no positive-capacity ratio at
        # all: it classifies each of the three edges and checks the loaded
        # zero-capacity load once, and the kernel loop adds one feasibility
        # test per edge.
        assert result.congestion is None
        assert result.work.rational_divisions_per_pass == 0
        assert result.work.exact_comparisons_per_pass == 4 + 3 + (3 + 1)
        assert result.work.logical_steps_per_call == (9 + 7 + 3) + 1 + 0 + 11


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
    result = compute_multicommodity_flow_profile(flow)
    assert result.all_demands_routed is True
    assert result.capacity_feasible is True
    assert result.congestion == q(1)
    assert all(row.slack == q(0) for row in result.edge_profiles)


def test_sparse_tensor_rejects_zero_duplicate_unknown_and_unsorted_cells() -> None:
    payload = shared_bottleneck_flow().model_dump(mode="json")

    zero = deepcopy(payload)
    zero["entries"][0]["amount"] = {"num": "0", "den": "1"}
    with multicommodity_validation_error():
        MulticommodityFlow.model_validate(zero)

    duplicate = deepcopy(payload)
    duplicate["entries"].append(deepcopy(duplicate["entries"][0]))
    with multicommodity_validation_error():
        MulticommodityFlow.model_validate(duplicate)

    undeclared = deepcopy(payload)
    undeclared["entries"][1]["commodity_id"] = "aa"
    with multicommodity_validation_error():
        MulticommodityFlow.model_validate(undeclared)

    unsorted = deepcopy(payload)
    unsorted["entries"] = list(reversed(unsorted["entries"]))
    with multicommodity_validation_error():
        MulticommodityFlow.model_validate(unsorted)


def test_low_commodity_networks_admit_vertices_up_to_the_cell_budget() -> None:
    # A 33-vertex graph with one edge and terminals 0->1 has tiny work and
    # output; the commodity-vertex cell budget (K*V <= 512) and the aggregate
    # result envelope bound admission, not an independent vertex ceiling.
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=33,
            edges=(CapacitatedEdge(source=0, target=1, capacity=q(2)),),
        ),
        commodities=(CommodityDemand(commodity_id="a", source=0, sink=1, demand=q(1)),),
        entries=(CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=q(1)),),
    )
    result = compute_multicommodity_flow_profile(flow)
    assert result.all_demands_routed is True
    assert result.capacity_feasible is True
    assert result.congestion == q(1, 2)
    assert len(result.divergences) == 33
    assert result.work.commodity_vertex_cells == 33
    assert result.work.exact_comparisons_per_pass == 33 + 3
    assert result.work.logical_steps_per_call == (3 + 3 + 1) + 1 + 1 + 36

    # Eight commodities over all 64 FlowGraph vertices reach exactly 512
    # returned cells; a ninth commodity exceeds the dense divergence budget.
    def wide_flow(commodity_count: int) -> MulticommodityFlow:
        return MulticommodityFlow(
            network=FlowGraph(
                vertex_count=64,
                edges=(CapacitatedEdge(source=0, target=1, capacity=q(1)),),
            ),
            commodities=tuple(
                CommodityDemand(
                    commodity_id=chr(ord("a") + index),
                    source=0,
                    sink=1,
                    demand=q(1),
                )
                for index in range(commodity_count)
            ),
        )

    full = compute_multicommodity_flow_profile(wide_flow(8))
    assert len(full.divergences) == 512
    assert full.work.commodity_vertex_cells == MAX_COMMODITY_VERTEX_CELLS

    with multicommodity_validation_error():
        wide_flow(9)


def test_large_exact_scalars_are_admitted_when_derived_digits_stay_bounded() -> None:
    # A 33-digit capacity performs constant work here and returns only a
    # 33-digit slack; operand-derived digit budgets, not a fixed input cap,
    # decide whether such exact scalars are admitted. Even on this one-edge
    # flow the single slack subtraction and the single congestion division
    # execute once because the kernel reuses the measured components of the
    # shared derivative scan.
    big_capacity = q(10**32)
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=2,
            edges=(CapacitatedEdge(source=0, target=1, capacity=big_capacity),),
        ),
        commodities=(CommodityDemand(commodity_id="a", source=0, sink=1, demand=q(1)),),
    )
    result = compute_multicommodity_flow_profile(flow)
    assert result.all_demands_routed is False
    assert result.capacity_feasible is True
    assert result.congestion == q(0)
    assert result.edge_profiles[0].slack == big_capacity
    assert result.work.sparse_entries == 0
    assert result.work.commodity_vertex_cells == 2
    assert result.work.edge_cells == 1
    assert result.work.rational_additions_per_pass == 1
    assert result.work.rational_negations_per_pass == 1
    assert result.work.rational_divisions_per_pass == 1
    assert result.work.exact_comparisons_per_pass == 5
    assert result.work.logical_steps_per_call == 8


@pytest.mark.scale
def test_operand_digit_budget_bounds_the_canonical_boundary() -> None:
    def unit_edge_amounts(
        amounts: tuple[CanonicalRational, ...],
    ) -> MulticommodityFlow:
        return MulticommodityFlow(
            network=FlowGraph(
                vertex_count=2,
                edges=(CapacitatedEdge(source=0, target=1, capacity=q(1)),),
            ),
            commodities=tuple(
                CommodityDemand(
                    commodity_id=chr(ord("a") + index), source=0, sink=1, demand=q(1)
                )
                for index in range(len(amounts))
            ),
            entries=tuple(
                CommodityEdgeFlow(
                    commodity_id=chr(ord("a") + index),
                    source=0,
                    target=1,
                    amount=amount,
                )
                for index, amount in enumerate(amounts)
            ),
        )

    # A lone amount is its own reduced sum: even at the canonical 32,768-digit
    # maximum, measured admission keeps it because every derived component
    # stays within that operand's own size.
    at_boundary = unit_edge_amounts((CanonicalRational(num="9" * 32_768, den="1"),))
    result = compute_multicommodity_flow_profile(at_boundary)
    assert result.capacity_feasible is False
    assert result.edge_profiles[0].load.num == "9" * 32_768
    assert result.congestion == CanonicalRational(num="9" * 32_768, den="1")

    # Two such amounts on one edge genuinely add: their exact load has
    # 32,769 digits, above the canonical cap, so the profile boundary fails
    # closed while the canonical tensor itself stays constructible.
    over_boundary = unit_edge_amounts(
        (
            CanonicalRational(num="9" * 32_768, den="1"),
            CanonicalRational(num="9" * 32_768, den="1"),
        )
    )
    MulticommodityFlowProfileRequest(flow=over_boundary)
    with pytest.raises(ValueError, match="canonical cap"):
        compute_multicommodity_flow_profile(over_boundary)


@pytest.mark.scale
def test_sole_operand_components_inherit_their_canonical_sides() -> None:
    # A two-vertex, one-edge tensor whose sole entry, matching capacity, and
    # demand are the same reduced rational with a 20,000-digit numerator and
    # denominator: every divergence and the load equal that already-canonical
    # operand, the slack is the exact zero rational, and the congestion is
    # exactly one. Admission tracks the contributing operand count instead of
    # charging nonexistent summation growth, so no 40,008-digit component is
    # implied and the tensor stays far below every bound.
    operand = CanonicalRational(num="9" * 20_000, den="1" + "0" * 19_999)
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=2,
            edges=(CapacitatedEdge(source=0, target=1, capacity=operand),),
        ),
        commodities=(
            CommodityDemand(commodity_id="a", source=0, sink=1, demand=operand),
        ),
        entries=(
            CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=operand),
        ),
    )
    assert derived_profile_digit_budget(flow) == 20_000
    result = compute_multicommodity_flow_profile(flow)
    assert result.all_demands_routed is True
    assert result.capacity_feasible is True
    assert result.congestion == q(1)
    assert result.edge_profiles[0].load == operand
    assert result.edge_profiles[0].slack == q(0)
    negative_operand = CanonicalRational(num="-" + "9" * 20_000, den="1" + "0" * 19_999)
    assert [row.divergence for row in result.divergences] == [
        operand,
        negative_operand,
    ]


@pytest.mark.scale
def test_rational_components_are_bounded_independently() -> None:
    # A reduced capacity whose numerator and denominator are EACH 20,000
    # digits is itself a valid canonical rational: its slack equals that
    # capacity exactly and its congestion is zero, so numerator and denominator
    # growth are tracked separately and the tensor is admitted even though the
    # summed component lengths exceed the per-component 32,768-digit cap.
    capacity = CanonicalRational(num="9" * 20_000, den="1" + "0" * 19_999)
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=2,
            edges=(CapacitatedEdge(source=0, target=1, capacity=capacity),),
        ),
        commodities=(CommodityDemand(commodity_id="a", source=0, sink=1, demand=q(1)),),
    )
    # With no entries the load is exactly zero: the slack equals the capacity,
    # the congestion is exactly zero, and no composition digit is added, so
    # the shared budget is just the larger capacity component.
    assert derived_profile_digit_budget(flow) == 20_000
    result = compute_multicommodity_flow_profile(flow)
    assert result.capacity_feasible is True
    assert result.congestion == q(0)
    assert result.edge_profiles[0].slack == capacity


@pytest.mark.scale
def test_per_component_digit_bounds_admit_unrelated_large_operands() -> None:
    # Loads and slacks are computed independently per edge and divergence
    # cells independently per commodity/vertex pair, so each component's
    # budget covers only the operands that can reach it. Two unrelated
    # 16,380-digit capacities stay admitted even though summing both would
    # imply a 32,770-digit bound above the canonical 32,768-digit cap.
    big_capacity = CanonicalRational(num="9" * 16_380, den="1")
    capacities_flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=3,
            edges=(
                CapacitatedEdge(source=0, target=1, capacity=big_capacity),
                CapacitatedEdge(source=1, target=2, capacity=big_capacity),
            ),
        ),
        commodities=(CommodityDemand(commodity_id="a", source=0, sink=2, demand=q(1)),),
    )
    # With no entries each load is exactly zero: its slack equals its
    # capacity, its congestion ratio is exactly zero, and no composition
    # digit is added, so the shared budget is just a capacity component.
    assert derived_profile_digit_budget(capacities_flow) == 16_380
    capacities_result = compute_multicommodity_flow_profile(capacities_flow)
    assert capacities_result.capacity_feasible is True
    assert capacities_result.congestion == q(0)
    assert [row.slack for row in capacities_result.edge_profiles] == [
        big_capacity,
        big_capacity,
    ]

    # The same independence holds between amounts on different edges: no
    # component ever sums both operands, so the paired tensor below stays
    # admitted although its aggregated operand digits exceed the cap.
    big_amount = CanonicalRational(num="9" * 16_380, den="1")
    amounts_flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=3,
            edges=(
                CapacitatedEdge(source=0, target=1, capacity=q(1)),
                CapacitatedEdge(source=1, target=2, capacity=q(1)),
            ),
        ),
        commodities=(
            CommodityDemand(commodity_id="a", source=0, sink=1, demand=q(1)),
            CommodityDemand(commodity_id="b", source=1, sink=2, demand=q(1)),
        ),
        entries=(
            CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=big_amount),
            CommodityEdgeFlow(commodity_id="b", source=1, target=2, amount=big_amount),
        ),
    )
    # With unrelated operands no derived value ever sums them: each component
    # measures at its own operand's size or one borrow digit beyond it.
    assert derived_profile_digit_budget(amounts_flow) == 16_380
    amounts_result = compute_multicommodity_flow_profile(amounts_flow)
    assert amounts_result.all_demands_routed is False
    assert amounts_result.edge_profiles[0].load == big_amount


def test_derived_quantities_admit_edges_without_a_fixed_ceiling() -> None:
    # FlowGraph owns the structural edge domain; this operation controls only
    # the ledger and result envelope. A one-commodity 129-edge network over 12
    # vertices executes 787 steps per pass with a tiny result, so no
    # independent edge ceiling may reject it.
    def many_edge_flow(vertex_count: int, edge_count: int) -> MulticommodityFlow:
        pairs = [
            (source, target)
            for source in range(vertex_count)
            for target in range(vertex_count)
            if source != target
        ][:edge_count]
        return MulticommodityFlow(
            network=FlowGraph(
                vertex_count=vertex_count,
                edges=tuple(
                    CapacitatedEdge(source=source, target=target, capacity=q(1))
                    for source, target in pairs
                ),
            ),
            commodities=(
                CommodityDemand(
                    commodity_id="a", source=0, sink=vertex_count - 1, demand=q(1)
                ),
            ),
        )

    result = compute_multicommodity_flow_profile(many_edge_flow(12, 129))
    assert result.all_demands_routed is False
    assert result.capacity_feasible is True
    assert result.congestion == q(0)
    assert len(result.edge_profiles) == 129
    assert result.work.edge_cells == 129
    assert result.work.rational_additions_per_pass == 129
    assert result.work.exact_comparisons_per_pass == 12 + 3 * 129
    assert result.work.logical_steps_per_call == 129 + 1 + 129 + 12 + 3 * 129

    # The ledger and returned rows inherit FlowGraph's own 512-edge maximum:
    # a full 512-edge network is admitted and accounted exactly.
    full = compute_multicommodity_flow_profile(many_edge_flow(64, 512))
    assert len(full.edge_profiles) == 512
    assert full.work.logical_steps_per_call == 512 + 1 + 512 + 64 + 3 * 512

    # Eight commodities over the same full 512-edge graph fill the dense
    # divergence budget alongside every edge: 512 cells plus three comparisons
    # per edge is exactly the published comparison envelope, and the widened
    # work fields must admit that worst case.
    crowded = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=64,
            edges=tuple(
                CapacitatedEdge(source=source, target=target, capacity=q(1))
                for source, target in [
                    (source, target)
                    for source in range(64)
                    for target in range(64)
                    if source != target
                ][:512]
            ),
        ),
        commodities=tuple(
            CommodityDemand(
                commodity_id=chr(ord("a") + index), source=0, sink=63, demand=q(1)
            )
            for index in range(8)
        ),
    )
    crowded_result = compute_multicommodity_flow_profile(crowded)
    assert crowded_result.capacity_feasible is True
    assert crowded_result.congestion == q(0)
    assert crowded_result.work.commodity_vertex_cells == MAX_COMMODITY_VERTEX_CELLS
    assert crowded_result.work.edge_cells == 512
    assert crowded_result.work.exact_comparisons_per_pass == (
        MAX_PROFILE_COMPARISONS_PER_PASS
    )
    assert crowded_result.work.logical_steps_per_call <= MAX_PROFILE_LOGICAL_STEPS

    with pytest.raises(ValidationError) as error:
        many_edge_flow(64, 513)
    assert error.value.errors()[0]["type"] == "too_long"


def test_entry_count_is_bounded_by_distinct_cells_not_a_fixed_ceiling() -> None:
    # Sparse entries are distinct commodity-by-edge cells, so a one-commodity
    # network with 129 sorted unit edges carries 129 admissible unit entries:
    # 13 divergence rows, 129 edge rows, and 1,175 steps per pass sit well
    # inside the published addition, comparison, work, and output envelopes.
    def filled_network(vertex_count: int, edge_count: int) -> MulticommodityFlow:
        pairs = [
            (source, target)
            for source in range(vertex_count)
            for target in range(vertex_count)
            if source != target
        ][:edge_count]
        return MulticommodityFlow(
            network=FlowGraph(
                vertex_count=vertex_count,
                edges=tuple(
                    CapacitatedEdge(source=source, target=target, capacity=q(1))
                    for source, target in pairs
                ),
            ),
            commodities=(
                CommodityDemand(
                    commodity_id="a", source=0, sink=vertex_count - 1, demand=q(1)
                ),
            ),
            entries=tuple(
                CommodityEdgeFlow(
                    commodity_id="a", source=source, target=target, amount=q(1)
                )
                for source, target in pairs
            ),
        )

    # Every amount shares denominator 1, so each bucket performs exactly one
    # fold per distinct bucket key: one per touched commodity/vertex cell and
    # one per loaded edge, counted independently of the kernel below.
    def unit_fold_additions(flow: MulticommodityFlow) -> int:
        touched_cells = {
            (entry.commodity_id, vertex)
            for entry in flow.entries
            for vertex in (entry.source, entry.target)
        }
        return len(touched_cells) + len({(e.source, e.target) for e in flow.entries})

    flow = filled_network(13, 129)
    result = compute_multicommodity_flow_profile(flow)
    assert result.capacity_feasible is True
    assert result.congestion == q(1)
    assert result.work.sparse_entries == 129
    assert len(result.edge_profiles) == 129
    assert all(row.load == q(1) for row in result.edge_profiles)
    assert result.work.rational_additions_per_pass == (
        3 * 129 + unit_fold_additions(flow) + 129
    )
    assert result.work.exact_comparisons_per_pass == 13 + 3 * 129
    assert result.work.logical_steps_per_call == (
        3 * 129 + unit_fold_additions(flow) + 129 + 1 + 129 + (13 + 3 * 129)
    )

    # The entry count inherits the derived cell maxima: one commodity over a
    # full 512-edge graph admits exactly 512 distinct entries.
    full_flow = filled_network(64, 512)
    full = compute_multicommodity_flow_profile(full_flow)
    assert full.work.sparse_entries == 512
    assert full.work.rational_additions_per_pass == (
        3 * 512 + unit_fold_additions(full_flow) + 512
    )


def test_sparse_scan_admits_commodities_over_a_full_edge_graph() -> None:
    # The kernel scans sparse entries and per-edge sums; it never materializes
    # a dense commodity-by-edge tensor, so five commodities over a full
    # 512-edge graph are admitted on the quantities actually consumed and
    # returned: 120 divergence rows, 512 small edge rows, 6,394 steps.
    def wide_network(commodity_count: int) -> MulticommodityFlow:
        pairs = [
            (source, target)
            for source in range(24)
            for target in range(24)
            if source != target
        ][:512]
        return MulticommodityFlow(
            network=FlowGraph(
                vertex_count=24,
                edges=tuple(
                    CapacitatedEdge(source=source, target=target, capacity=q(1))
                    for source, target in pairs
                ),
            ),
            commodities=tuple(
                CommodityDemand(
                    commodity_id=f"c{index:04d}", source=0, sink=23, demand=q(1)
                )
                for index in range(commodity_count)
            ),
        )

    result = compute_multicommodity_flow_profile(wide_network(5))
    assert result.all_demands_routed is False
    assert result.capacity_feasible is True
    assert result.congestion == q(0)
    assert len(result.divergences) == 120
    assert len(result.edge_profiles) == 512
    assert result.work.rational_additions_per_pass == 512
    assert result.work.rational_negations_per_pass == 5
    assert result.work.rational_divisions_per_pass == 512
    assert result.work.exact_comparisons_per_pass == 120 + 3 * 512
    assert result.work.logical_steps_per_call == 512 + 5 + 512 + 120 + 3 * 512


def test_cell_budgets_admit_commodities_without_a_fixed_ceiling() -> None:
    # Commodity count is bounded by the derived commodity-vertex cell budget
    # rather than an independent fixed ceiling: 17 commodities over two
    # vertices occupy only 34 cells of constant size.
    def dense_commodities(commodity_count: int) -> MulticommodityFlow:
        return MulticommodityFlow(
            network=FlowGraph(
                vertex_count=2,
                edges=(CapacitatedEdge(source=0, target=1, capacity=q(1)),),
            ),
            commodities=tuple(
                CommodityDemand(
                    commodity_id=f"c{index:04d}", source=0, sink=1, demand=q(1)
                )
                for index in range(commodity_count)
            ),
        )

    unrouted = compute_multicommodity_flow_profile(dense_commodities(17))
    assert unrouted.all_demands_routed is False
    assert unrouted.capacity_feasible is True
    assert unrouted.congestion == q(0)
    assert len(unrouted.divergences) == 34
    assert unrouted.work.commodity_vertex_cells == 34
    assert unrouted.work.rational_negations_per_pass == 17
    assert unrouted.work.logical_steps_per_call == 1 + 17 + 1 + 34 + 3

    # Each commodity occupies at least two distinct commodity-vertex cells,
    # so exactly half of the 512-cell divergence budget, 256 commodities, is
    # admitted over two vertices; one more exceeds the dense table.
    full = compute_multicommodity_flow_profile(dense_commodities(256))
    assert len(full.divergences) == MAX_COMMODITY_VERTEX_CELLS
    assert full.work.rational_negations_per_pass == 256
    assert full.work.logical_steps_per_call == 1 + 256 + 1 + 512 + 3

    with multicommodity_validation_error():
        dense_commodities(257)


@pytest.mark.scale
def test_result_envelope_prices_rows_at_their_actual_sides() -> None:
    from itertools import combinations

    def comb_edge_tensor(
        capacity: CanonicalRational, edge_count: int
    ) -> MulticommodityFlow:
        edges = tuple(
            CapacitatedEdge(source=source, target=target, capacity=capacity)
            for source, target in combinations(range(32), 2)
        )[:edge_count]
        return MulticommodityFlow(
            network=FlowGraph(vertex_count=32, edges=edges),
            commodities=tuple(
                CommodityDemand(
                    commodity_id=f"c{index:02d}", source=0, sink=31, demand=q(1)
                )
                for index in range(4)
            ),
        )

    def comb_amount_tensor(amount_digits: int) -> MulticommodityFlow:
        pairs = list(combinations(range(32), 2))[:128]
        amount = CanonicalRational(num="9" * amount_digits, den="1")
        return MulticommodityFlow(
            network=FlowGraph(
                vertex_count=32,
                edges=tuple(
                    CapacitatedEdge(source=source, target=target, capacity=q(1))
                    for source, target in pairs
                ),
            ),
            commodities=(
                CommodityDemand(commodity_id="a", source=0, sink=31, demand=q(1)),
            ),
            entries=tuple(
                CommodityEdgeFlow(
                    commodity_id="a", source=source, target=target, amount=amount
                )
                for source, target in pairs
            ),
        )

    # With no entries each load is exactly zero and each slack equals its
    # capacity: 100 capacities with 32,000-digit numerators echo about
    # 3.2 MB and repeat them once as slacks, about 6.4 MB total, so the
    # request stays inside the aggregate envelope. Pricing the zero loads at
    # the slack bound would have inflated this above 8 MiB and rejected it.
    admitted = comb_edge_tensor(CanonicalRational(num="9" * 32_000, den="1"), 100)
    assert derived_profile_digit_budget(admitted) == 32_000

    # Unit-capacity edges carrying 22,000-digit amounts make the echoed
    # entries, divergence cells, loads, slacks, and congestion genuinely
    # exceed 8 MiB together. Wire parsing remains structural; execution
    # admission rejects the profile before result construction.
    flow = comb_amount_tensor(22_000)
    MulticommodityFlowProfileRequest(flow=flow)
    with pytest.raises(ValueError, match="aggregate result bound"):
        compute_multicommodity_flow_profile(flow)


@pytest.mark.scale
def test_congestion_bound_uses_the_capacity_denominator() -> None:
    # A unit load over capacity 1/D reduces to exactly D: cross-multiplication
    # grows the congestion numerator by the capacity denominator's 4,000
    # digits plus one additive digit, not by its one-digit numerator.
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=2,
            edges=(
                CapacitatedEdge(
                    source=0,
                    target=1,
                    capacity=CanonicalRational(num="1", den="5" * 4_000),
                ),
            ),
        ),
        commodities=(CommodityDemand(commodity_id="a", source=0, sink=1, demand=q(1)),),
        entries=(CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=q(1)),),
    )
    scan = measured_profile_components(flow)
    assert scan.congestion_bound == (4_000, 1)
    assert max(scan.load_bounds.values()) == (1, 1)
    assert scan.slack_bounds[(0, 1)] == (4_000, 4_000)


def oversized_ratio_tensor() -> MulticommodityFlow:
    # A load a/b over a capacity c/d whose reduced quotient has both sides
    # far above the 32,768-digit canonical cap: the numerators are 30,000
    # digits, the denominators 5,001 digits, and c*b - a*d stays below d, so
    # every returned component -- divergences, load, slack -- remains inside
    # the cap while only the congestion quotient exceeds it.
    b = 10**5_000 + 3
    d = 10**5_000 + 7
    c = 10**30_000 + 9
    a = (c * b) // d

    def rational(value: Fraction) -> CanonicalRational:
        return CanonicalRational.from_fraction(value)

    load = Fraction(a, b)
    capacity = Fraction(c, d)
    return MulticommodityFlow(
        network=FlowGraph(
            vertex_count=2,
            edges=(CapacitatedEdge(source=0, target=1, capacity=rational(capacity)),),
        ),
        commodities=(
            CommodityDemand(commodity_id="a", source=0, sink=1, demand=rational(load)),
        ),
        entries=(
            CommodityEdgeFlow(
                commodity_id="a", source=0, target=1, amount=rational(load)
            ),
        ),
    )


@pytest.mark.scale
def test_null_congestion_admits_ratios_the_result_omits() -> None:
    # A positive-capacity edge whose reduced load/capacity quotient exceeds
    # the canonical cap stands beside an unrelated loaded zero-capacity edge,
    # so the public result's congestion is null exactly as its contract
    # defines: the oversized ratio is a value the result omits, and admission
    # must not reject a representable profile because of it. The shared
    # derivative scan therefore divides no ratio at all, prices no congestion
    # row beyond the null field's bytes, and excludes it from the budget.
    from jacobian.canonical import encode_strict_json
    from jacobian.math.graphs.flows.multicommodity._models import (
        _RATIONAL_JSON_OVERHEAD_BYTES,
        derived_profile_digit_budget,
    )

    b = 10**5_000 + 3
    d = 10**5_000 + 7
    c = 10**30_000 + 9
    a = (c * b) // d

    def rational(value: Fraction) -> CanonicalRational:
        return CanonicalRational.from_fraction(value)

    capacity = Fraction(c, d)
    load = Fraction(a, b)
    ratio = load / capacity
    assert len(format_canonical_integer(abs(ratio.numerator))) > (
        MAX_CANONICAL_RATIONAL_DIGITS
    )
    assert len(format_canonical_integer(load.numerator)) < MAX_CANONICAL_RATIONAL_DIGITS
    assert len(format_canonical_integer(ratio.denominator)) > (
        MAX_CANONICAL_RATIONAL_DIGITS
    )
    # The reserved overhead must cover the serialized null congestion field,
    # otherwise null-priced envelopes would underprice their results.
    assert (
        len(encode_strict_json({"congestion": None})) <= _RATIONAL_JSON_OVERHEAD_BYTES
    )
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=3,
            edges=(
                CapacitatedEdge(source=0, target=1, capacity=rational(capacity)),
                CapacitatedEdge(
                    source=1, target=2, capacity=CanonicalRational(num="0", den="1")
                ),
            ),
        ),
        commodities=(
            CommodityDemand(commodity_id="a", source=0, sink=1, demand=rational(load)),
            CommodityDemand(commodity_id="b", source=1, sink=2, demand=q(1)),
        ),
        entries=(
            CommodityEdgeFlow(
                commodity_id="a", source=0, target=1, amount=rational(load)
            ),
            CommodityEdgeFlow(commodity_id="b", source=1, target=2, amount=q(1)),
        ),
    )
    # Every priced component is the operands' own size; the omitted ratio's
    # 35,001-digit sides never enter the budget.
    assert derived_profile_digit_budget(flow) == 30_000
    result = compute_multicommodity_flow_profile(flow)
    via_request = _run_multicommodity_flow_profile(
        MulticommodityFlowProfileRequest(flow=flow)
    )
    assert result == via_request
    assert result.all_demands_routed is True
    assert result.capacity_feasible is False
    assert result.congestion is None
    assert [row.load for row in result.edge_profiles] == [
        rational(load),
        q(1),
    ]
    # Two sparse entries fold one denominator into each of their two
    # divergence cells plus two edge buckets (six folds), two slack
    # subtractions run, and the scan classifies each edge once and checks
    # the loaded zero-capacity load once while dividing nothing; the kernel
    # loop adds one feasibility test per edge.
    assert result.work.sparse_entries == 2
    assert result.work.rational_additions_per_pass == 3 * 2 + 6 + 2
    assert result.work.rational_divisions_per_pass == 0
    assert result.work.exact_comparisons_per_pass == 6 + 2 + (2 + 1)
    assert result.work.logical_steps_per_call == 14 + 2 + 0 + 11


@pytest.mark.scale
def test_returned_congestion_ratio_is_still_capped() -> None:
    # Without a loaded zero-capacity edge the same oversized quotient is the
    # value the result actually returns under "congestion", so the canonical
    # cap still applies to it fail-closed on both surfaces: narrowing the
    # scan to returned values did not widen the returned-value contract.
    flow = oversized_ratio_tensor()
    with pytest.raises(ValueError, match="canonical cap"):
        compute_multicommodity_flow_profile(flow)


@pytest.mark.scale
def test_shared_denominator_sums_stay_at_operand_size() -> None:
    # Two commodities carry 1/D and (D-1)/D over one unit-capacity edge with
    # matching demands: the exact load is 1, the slack is exactly zero, the
    # congestion is exactly one, and every divergence stays within D's own
    # 20,000-digit sides -- the shared denominator prevents any denominator
    # growth, so no summed digit-count bound rejects this small result.
    denominator = "5" * 20_000
    first = CanonicalRational(num="1", den=denominator)
    second = CanonicalRational(num="5" * 19_999 + "4", den=denominator)
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=2,
            edges=(CapacitatedEdge(source=0, target=1, capacity=q(1)),),
        ),
        commodities=(
            CommodityDemand(commodity_id="a", source=0, sink=1, demand=first),
            CommodityDemand(commodity_id="b", source=0, sink=1, demand=second),
        ),
        entries=(
            CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=first),
            CommodityEdgeFlow(commodity_id="b", source=0, target=1, amount=second),
        ),
    )
    assert derived_profile_digit_budget(flow) == 20_000
    result = compute_multicommodity_flow_profile(flow)
    assert result.all_demands_routed is True
    assert result.capacity_feasible is True
    assert result.congestion == q(1)
    assert result.edge_profiles[0].load == q(1)
    assert result.edge_profiles[0].slack == q(0)


@pytest.mark.scale
def test_cancelling_amounts_are_admitted_regardless_of_entry_order() -> None:
    # Canonical entry order processes both edges leaving vertex 0 before the
    # incoming ones, so the partial divergence at vertex 0 is 2A. The
    # canonical cap applies to completed components only: every final
    # divergence cancels to at most one digit, each large edge has exactly
    # zero slack and unit congestion, and the tensor is admitted.
    big = CanonicalRational(num="9" * 32_768, den="1")
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=3,
            edges=(
                CapacitatedEdge(source=0, target=1, capacity=big),
                CapacitatedEdge(source=0, target=2, capacity=big),
                CapacitatedEdge(source=1, target=0, capacity=big),
                CapacitatedEdge(source=1, target=2, capacity=q(1)),
                CapacitatedEdge(source=2, target=0, capacity=big),
            ),
        ),
        commodities=(CommodityDemand(commodity_id="a", source=0, sink=2, demand=q(1)),),
        entries=(
            CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=big),
            CommodityEdgeFlow(commodity_id="a", source=0, target=2, amount=big),
            CommodityEdgeFlow(commodity_id="a", source=1, target=0, amount=big),
            CommodityEdgeFlow(commodity_id="a", source=1, target=2, amount=q(1)),
            CommodityEdgeFlow(commodity_id="a", source=2, target=0, amount=big),
        ),
    )
    result = compute_multicommodity_flow_profile(flow)
    assert len(result.divergences) == 3
    assert {row.divergence for row in result.divergences} <= {q(1), q(-1), q(0)}
    assert [row.slack for row in result.edge_profiles[:4]] == [q(0)] * 4
    assert result.edge_profiles[4].slack == q(0)
    assert result.congestion == q(1)
    assert result.capacity_feasible is True


@pytest.mark.scale
def test_coprime_denominator_flood_fails_closed() -> None:
    # Two unit-fraction entries whose distinct near-cap denominators are
    # coprime complete to a roughly 32,777-digit load denominator above the
    # canonical cap: their single fold stays inside the fold-intermediate
    # budget, and admission fails closed on the measured completed load
    # instead of admitting the huge component.
    network = FlowGraph(
        vertex_count=2,
        edges=(CapacitatedEdge(source=0, target=1, capacity=q(1)),),
    )
    commodities = (
        CommodityDemand(commodity_id="a", source=0, sink=1, demand=q(1)),
        CommodityDemand(commodity_id="b", source=0, sink=1, demand=q(1)),
    )
    entries = (
        CommodityEdgeFlow(
            commodity_id="a",
            source=0,
            target=1,
            amount=CanonicalRational(num="1", den="3" + "0" * 19_999),
        ),
        CommodityEdgeFlow(
            commodity_id="b",
            source=0,
            target=1,
            amount=CanonicalRational(num="1", den="7" + "0" * 12_775 + "1"),
        ),
    )
    flood = MulticommodityFlow(
        network=network, commodities=commodities, entries=entries
    )
    MulticommodityFlowProfileRequest(flow=flood)
    with pytest.raises(ValueError, match="canonical cap"):
        compute_multicommodity_flow_profile(flood)


def oversized_echo_flow() -> MulticommodityFlow:
    # One commodity carries 270 lone 32,000-digit entries on distinct edges:
    # every derived component is a single operand inside the canonical cap,
    # so only the serialized echo -- about 8.7 MB of numerator digits --
    # exhausts the 8 MiB aggregate result envelope.
    vertex_count = 17
    pairs = [
        (source, target)
        for source in range(vertex_count)
        for target in range(vertex_count)
        if source != target
    ][:270]
    big = CanonicalRational(num="9" * 32_000, den="1")
    return MulticommodityFlow(
        network=FlowGraph(
            vertex_count=vertex_count,
            edges=tuple(
                CapacitatedEdge(source=source, target=target, capacity=q(1))
                for source, target in pairs
            ),
        ),
        commodities=(
            CommodityDemand(commodity_id="a", source=0, sink=16, demand=q(1)),
        ),
        entries=tuple(
            CommodityEdgeFlow(
                commodity_id="a", source=source, target=target, amount=big
            )
            for source, target in pairs
        ),
    )


@pytest.mark.scale
def test_oversized_source_is_rejected_before_the_component_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.graphs.flows.multicommodity import _models

    executed: list[MulticommodityFlow] = []
    original_scan = _models._component_sums_with_folds

    def scan_spy(flow: MulticommodityFlow) -> object:
        executed.append(flow)
        return original_scan(flow)

    monkeypatch.setattr(_models, "_component_sums_with_folds", scan_spy)
    # Request parsing remains structural. Execution measures the echoed source
    # before its own component scan and therefore never starts the scan.
    flow = oversized_echo_flow()
    MulticommodityFlowProfileRequest(flow=flow)
    assert executed == []
    # A native call is rejected by the same preflight inside the kernel's
    # own admission, likewise before any rational arithmetic.
    with pytest.raises(ValueError, match="aggregate result bound"):
        compute_multicommodity_flow_profile(flow)
    assert executed == []


@pytest.mark.scale
def test_oversized_source_rejection_precedes_a_doomed_exact_scan() -> None:
    # Same oversized echo, but two commodities fold coprime near-cap
    # denominators into one shared edge bucket whose sum would abort the
    # component scan with a canonical-cap error. The envelope preflight must
    # win: the request fails for its result size before any exact arithmetic.
    vertex_count = 17
    bulk_pairs = [
        (source, target)
        for source in range(vertex_count)
        for target in range(vertex_count)
        if source != target and (source, target) != (0, 1)
    ][:268]
    big = CanonicalRational(num="9" * 32_000, den="1")
    doomed_echo = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=vertex_count,
            edges=tuple(
                CapacitatedEdge(source=source, target=target, capacity=q(1))
                for source, target in [(0, 1), *bulk_pairs]
            ),
        ),
        commodities=(
            CommodityDemand(commodity_id="a", source=0, sink=16, demand=q(1)),
            CommodityDemand(commodity_id="b", source=0, sink=1, demand=q(1)),
        ),
        entries=(
            CommodityEdgeFlow(
                commodity_id="a",
                source=0,
                target=1,
                amount=CanonicalRational(num="1", den="3" + "0" * 19_999),
            ),
            *(
                CommodityEdgeFlow(
                    commodity_id="a", source=source, target=target, amount=big
                )
                for source, target in bulk_pairs
            ),
            CommodityEdgeFlow(
                commodity_id="b",
                source=0,
                target=1,
                amount=CanonicalRational(num="1", den="7" + "0" * 12_775 + "1"),
            ),
        ),
    )
    MulticommodityFlowProfileRequest(flow=doomed_echo)
    with pytest.raises(ValueError, match="aggregate result bound"):
        compute_multicommodity_flow_profile(doomed_echo)


def test_ledger_charges_every_performed_bucket_fold_addition() -> None:
    # Five entries carry four pairwise-coprime denominators into buckets that
    # share cells and one edge, so the denominator-fold pass performs one
    # fraction addition per distinct (bucket, denominator) pair -- fourteen
    # here -- beyond the fifteen bucketed numerator additions and four slack
    # subtractions. The reference count below re-derives the bucket shapes
    # from the entry list itself, so the ledger can only match by charging
    # exactly the arithmetic the scan executes.
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
        commodities=(
            CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(1)),
            CommodityDemand(commodity_id="b", source=0, sink=1, demand=q(1)),
        ),
        entries=(
            CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=q(1, 2)),
            CommodityEdgeFlow(commodity_id="a", source=0, target=2, amount=q(1, 3)),
            CommodityEdgeFlow(commodity_id="a", source=1, target=3, amount=q(1, 5)),
            CommodityEdgeFlow(commodity_id="a", source=2, target=3, amount=q(1, 7)),
            CommodityEdgeFlow(commodity_id="b", source=0, target=1, amount=q(1, 2)),
        ),
    )

    def bucket_fold_additions(tensor: MulticommodityFlow) -> int:
        cell_denominators: dict[tuple[str, int], set[int]] = defaultdict(set)
        edge_denominators: dict[tuple[int, int], set[int]] = defaultdict(set)
        for entry in tensor.entries:
            denominator = entry.amount.as_fraction().denominator
            cell_denominators[(entry.commodity_id, entry.source)].add(denominator)
            cell_denominators[(entry.commodity_id, entry.target)].add(denominator)
            edge_denominators[(entry.source, entry.target)].add(denominator)
        cell_folds = sum(len(sides) for sides in cell_denominators.values())
        edge_folds = sum(len(sides) for sides in edge_denominators.values())
        return cell_folds + edge_folds

    result = compute_multicommodity_flow_profile(flow)
    folds = bucket_fold_additions(flow)
    assert folds == 14
    assert result.work.rational_additions_per_pass == 3 * 5 + folds + 4
    assert result.work.rational_negations_per_pass == 2
    assert result.work.rational_divisions_per_pass == 4
    assert result.work.exact_comparisons_per_pass == 8 + 3 * 4
    assert result.work.logical_steps_per_call == 33 + 2 + 4 + 20
    # The exact values stay the known answers even though the fold order
    # follows ascending denominator bit length rather than entry order.
    divergence = {
        (row.commodity_id, row.vertex): row.divergence.as_fraction()
        for row in result.divergences
    }
    assert divergence[("a", 0)] == Fraction(1, 2) + Fraction(1, 3)
    assert divergence[("b", 0)] == Fraction(1, 2)
    assert result.edge_profiles[0].load.as_fraction() == Fraction(1)


@pytest.mark.scale
def test_fold_intermediates_admit_coprime_sums_with_small_components() -> None:
    # One edge, four commodities carrying 1/p, 1/q, (p-2)/(2p), (q-2)/(2q)
    # with p and q coprime 20,000-digit odd integers: denominator sorting
    # folds 1/p + 1/q first, whose reduced intermediate has the roughly
    # 40,000-digit denominator p*q. That transient exceeds the 32,768-digit
    # canonical cap but stays inside the derived fold-intermediate budget,
    # and each completed sum is small -- the load is exactly one because
    # each corresponding pair sums to exactly one half. Fold order must
    # therefore not reject this tensor.
    p = 5 * 10**19_999 + 3

    def amount(numerator: int, denominator: int) -> CanonicalRational:
        return CanonicalRational(
            num=format_canonical_integer(numerator),
            den=format_canonical_integer(denominator),
        )

    commodity_amounts = (
        amount(1, p),
        amount(1, p + 2),
        amount(p - 2, 2 * p),
        amount(p, 2 * (p + 2)),
    )
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=2,
            edges=(CapacitatedEdge(source=0, target=1, capacity=q(1)),),
        ),
        commodities=tuple(
            CommodityDemand(
                commodity_id=chr(ord("a") + index),
                source=0,
                sink=1,
                demand=amount,
            )
            for index, amount in enumerate(commodity_amounts)
        ),
        entries=tuple(
            CommodityEdgeFlow(
                commodity_id=chr(ord("a") + index),
                source=0,
                target=1,
                amount=amount,
            )
            for index, amount in enumerate(commodity_amounts)
        ),
    )
    result = compute_multicommodity_flow_profile(flow)
    assert result.all_demands_routed is True
    assert result.capacity_feasible is True
    assert result.congestion == q(1)
    assert result.edge_profiles[0].load == q(1)
    assert result.edge_profiles[0].slack == q(0)
    assert [row.divergence for row in result.divergences] == [
        signed
        for amount in commodity_amounts
        for signed in (
            amount,
            CanonicalRational.from_fraction(-amount.as_fraction()),
        )
    ]


@pytest.mark.scale
def test_folding_intermediates_cancel_beneath_the_completed_cap() -> None:
    # One edge carries 1/p and 1/q alongside (p-2)/(2p) and (q-2)/(2q) for
    # coprime 20,000-digit odd p and q, with matching demands. Folding the
    # smallest denominators first forms the temporary (p+q)/(pq) whose
    # denominator has about 40,000 digits -- above the canonical cap but
    # within the separately derived fold-intermediate budget -- and the
    # later terms cancel each pair to exactly one half, so the completed
    # load is exactly 1 while every divergence stays input-sized.
    p_digits = "1" + "0" * 19_998 + "1"
    q_digits = "3" + "0" * 19_998 + "1"
    first = CanonicalRational(num="1", den=p_digits)
    second = CanonicalRational(num="1", den=q_digits)
    half_of_p = CanonicalRational(num="9" * 19_999, den="2" + "0" * 19_998 + "2")
    half_of_q = CanonicalRational(num="2" + "9" * 19_999, den="6" + "0" * 19_998 + "2")
    flow = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=2,
            edges=(CapacitatedEdge(source=0, target=1, capacity=q(1)),),
        ),
        commodities=(
            CommodityDemand(commodity_id="a", source=0, sink=1, demand=first),
            CommodityDemand(commodity_id="b", source=0, sink=1, demand=second),
            CommodityDemand(commodity_id="c", source=0, sink=1, demand=half_of_p),
            CommodityDemand(commodity_id="d", source=0, sink=1, demand=half_of_q),
        ),
        entries=(
            CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=first),
            CommodityEdgeFlow(commodity_id="b", source=0, target=1, amount=second),
            CommodityEdgeFlow(commodity_id="c", source=0, target=1, amount=half_of_p),
            CommodityEdgeFlow(commodity_id="d", source=0, target=1, amount=half_of_q),
        ),
    )
    assert derived_profile_digit_budget(flow) == 20_000
    result = compute_multicommodity_flow_profile(flow)
    assert result.all_demands_routed is True
    assert result.capacity_feasible is True
    assert result.congestion == q(1)
    assert result.edge_profiles[0].load == q(1)
    assert result.edge_profiles[0].slack == q(0)
    divergence = {
        (row.commodity_id, row.vertex): row.divergence for row in result.divergences
    }
    assert divergence[("c", 0)] == half_of_p
    assert divergence[("d", 1)].as_fraction() == -half_of_q.as_fraction()
    # Ledger: twelve entry additions, eight single-denominator divergence
    # folds plus four distinct edge-bucket denominator folds, one slack.
    assert result.work.sparse_entries == 4
    assert result.work.commodity_vertex_cells == 8
    assert result.work.rational_additions_per_pass == 3 * 4 + (8 + 4) + 1
    assert result.work.exact_comparisons_per_pass == 8 + 3
    assert result.work.logical_steps_per_call == 25 + 4 + 1 + 11


@pytest.mark.scale
def test_folds_are_bounded_by_the_intermediate_budget_not_the_component_cap() -> None:
    # Three pairwise-coprime 24,000-digit denominators meet in one divergence
    # cell (+1/p - 1/q - 1/r, sources minus sinks). Any pair fold keeps its
    # full product denominator -- about 48,000 digits, far above the
    # canonical cap yet inside the fold-intermediate budget -- and the third
    # fold's predicted unreduced product crosses that budget, so the request
    # fails closed before the cross-denominator arithmetic runs. Pairwise
    # coprimality: 3p-q = 2, r - 5p = -2, and r - q = 2p share no odd factor
    # with any denominator.
    p_digits = "1" + "0" * 23_998 + "1"
    q_digits = "3" + "0" * 23_998 + "1"
    r_digits = "5" + "0" * 23_998 + "3"
    cancelling_cell = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=3,
            edges=(
                CapacitatedEdge(source=0, target=1, capacity=q(1)),
                CapacitatedEdge(source=1, target=0, capacity=q(1)),
                CapacitatedEdge(source=2, target=0, capacity=q(1)),
            ),
        ),
        commodities=(CommodityDemand(commodity_id="a", source=0, sink=1, demand=q(1)),),
        entries=(
            CommodityEdgeFlow(
                commodity_id="a",
                source=0,
                target=1,
                amount=CanonicalRational(num="1", den=p_digits),
            ),
            CommodityEdgeFlow(
                commodity_id="a",
                source=1,
                target=0,
                amount=CanonicalRational(num="1", den=q_digits),
            ),
            CommodityEdgeFlow(
                commodity_id="a",
                source=2,
                target=0,
                amount=CanonicalRational(num="1", den=r_digits),
            ),
        ),
    )
    MulticommodityFlowProfileRequest(flow=cancelling_cell)
    with pytest.raises(ValueError, match="fold-intermediate budget"):
        compute_multicommodity_flow_profile(cancelling_cell)


@pytest.mark.scale
def test_near_cap_coprime_growth_aborts_within_budget_sized_arithmetic() -> None:
    # The exhaustion shape behind the finding: distinct near-cap coprime
    # denominators folded into one divergence cell (+1/p - 1/q - 1/r,
    # sources minus sinks of one commodity) grow a monotone accumulator
    # whose digit mass tracks no single operand. Because every fold is
    # pre-checked against the derived budget from measured operand sides,
    # the third fold is refused before its roughly 96,000-digit product is
    # ever constructed: admission never builds a fraction larger than twice
    # the canonical cap, whatever the request's total digit mass, so
    # request validation cannot be driven into multi-million-digit rational
    # arithmetic by a sub-envelope tensor. Pairwise coprimality as above.
    p_digits = "1" + "0" * 31_998 + "1"
    q_digits = "3" + "0" * 31_998 + "1"
    r_digits = "5" + "0" * 31_998 + "3"
    near_cap_growth = MulticommodityFlow(
        network=FlowGraph(
            vertex_count=3,
            edges=(
                CapacitatedEdge(source=0, target=1, capacity=q(1)),
                CapacitatedEdge(source=1, target=0, capacity=q(1)),
                CapacitatedEdge(source=2, target=0, capacity=q(1)),
            ),
        ),
        commodities=(CommodityDemand(commodity_id="a", source=0, sink=1, demand=q(1)),),
        entries=(
            CommodityEdgeFlow(
                commodity_id="a",
                source=0,
                target=1,
                amount=CanonicalRational(num="1", den=p_digits),
            ),
            CommodityEdgeFlow(
                commodity_id="a",
                source=1,
                target=0,
                amount=CanonicalRational(num="1", den=q_digits),
            ),
            CommodityEdgeFlow(
                commodity_id="a",
                source=2,
                target=0,
                amount=CanonicalRational(num="1", den=r_digits),
            ),
        ),
    )
    MulticommodityFlowProfileRequest(flow=near_cap_growth)
    with pytest.raises(ValueError, match="fold-intermediate budget"):
        compute_multicommodity_flow_profile(near_cap_growth)


def test_result_round_trip_and_validation_remains_structural_only() -> None:
    result = compute_multicommodity_flow_profile(shared_bottleneck_flow())
    payload = result.model_dump(mode="json")

    assert MulticommodityFlowProfileResult.model_validate(payload) == result

    forged_load = deepcopy(payload)
    forged_load["edge_profiles"][2]["load"] = {"num": "2", "den": "1"}
    assert MulticommodityFlowProfileResult.model_validate(forged_load)

    forged_source = deepcopy(payload)
    forged_source["flow"]["commodities"][1]["demand"] = {"num": "1", "den": "1"}
    assert MulticommodityFlowProfileResult.model_validate(forged_source)

    forged_work = deepcopy(payload)
    forged_work["work"]["logical_steps_per_call"] = 1
    assert MulticommodityFlowProfileResult.model_validate(forged_work)
