"""Contract tests for exact bounded multicommodity-flow profiles."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.flow._models import CapacitatedEdge, FlowGraph
from jacobian.math.graphs.multicommodity_flow._models import (
    MAX_COMMODITY_VERTEX_CELLS,
    CommodityDemand,
    CommodityEdgeFlow,
    MulticommodityFlow,
    MulticommodityFlowProfileRequest,
    MulticommodityFlowProfileResult,
    derived_profile_digit_budget,
)
from jacobian.math.graphs.multicommodity_flow._operations import (
    _run_multicommodity_flow_profile,
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


def test_native_api_accepts_the_canonical_flow_value_directly() -> None:
    flow = shared_bottleneck_flow()

    native = compute_multicommodity_flow_profile(flow)
    via_request = _run_multicommodity_flow_profile(
        MulticommodityFlowProfileRequest(flow=flow)
    )

    assert native == via_request
    assert native.flow == flow
    assert native.congestion == q(1)


def test_exact_shared_bottleneck_profile_replays_its_source() -> None:
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
    # A zero-capacity edge still compares load to capacity and capacity to zero,
    # then checks whether its load is positive; no ratio division is attempted.
    assert result.work.rational_divisions_per_pass == 0
    assert result.work.exact_comparisons_per_pass == 5
    assert result.work.logical_steps_per_call == 20


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
    assert result.work.logical_steps_per_call == 74


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
        # Both positive-capacity ratios are divided and compared against the
        # running maximum even when a preceding zero-capacity edge has already
        # made the congestion value null.
        assert result.congestion is None
        assert result.work.rational_divisions_per_pass == 2
        assert result.work.exact_comparisons_per_pass == 4 + 3 * 3
        assert result.work.logical_steps_per_call == 56


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
    assert result.work.logical_steps_per_call == 2 * (4 + 1 + 1 + 36)

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

    with pytest.raises(ValidationError, match="commodity-by-vertex"):
        wide_flow(9)


def test_large_exact_scalars_are_admitted_when_derived_digits_stay_bounded() -> None:
    # A 33-digit capacity performs constant work here and returns only a
    # 33-digit slack; operand-derived digit budgets, not a fixed input cap,
    # decide whether such exact scalars are admitted.
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
    assert result.work.logical_steps_per_call == 16


def test_operand_digit_budget_bounds_the_canonical_boundary() -> None:
    def unit_edge_amount(amount: CanonicalRational) -> MulticommodityFlow:
        return MulticommodityFlow(
            network=FlowGraph(
                vertex_count=2,
                edges=(CapacitatedEdge(source=0, target=1, capacity=q(1)),),
            ),
            commodities=(
                CommodityDemand(commodity_id="a", source=0, sink=1, demand=q(1)),
            ),
            entries=(
                CommodityEdgeFlow(commodity_id="a", source=0, target=1, amount=amount),
            ),
        )

    # One 32,757-digit amount numerator, its one-digit denominator, the eight
    # sum carry digits, and one subtraction-composition digit reach exactly
    # the canonical 32,768-digit component cap.
    at_boundary = unit_edge_amount(CanonicalRational(num="9" * 32_757, den="1"))
    result = compute_multicommodity_flow_profile(at_boundary)
    assert result.capacity_feasible is False
    assert result.edge_profiles[0].load.num == "9" * 32_757
    assert result.congestion == CanonicalRational(num="9" * 32_757, den="1")

    beyond_boundary = CanonicalRational(num="9" * 32_758, den="1")
    with pytest.raises(ValidationError, match="canonical cap"):
        unit_edge_amount(beyond_boundary)


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
    assert derived_profile_digit_budget(amounts_flow) == 8 + 16_381 + 2
    amounts_result = compute_multicommodity_flow_profile(amounts_flow)
    assert amounts_result.all_demands_routed is False
    assert amounts_result.edge_profiles[0].load == big_amount


def test_derived_quantities_admit_edges_without_a_fixed_ceiling() -> None:
    # FlowGraph owns the structural edge domain; this operation controls only
    # the derived commodity-edge cells, ledger, and result envelope. A
    # one-commodity 129-edge network over 12 vertices executes 658 steps per
    # pass with a tiny result, so no independent edge ceiling may reject it.
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
    assert result.work.logical_steps_per_call == 2 * (129 + 1 + 129 + 12 + 387)

    # The ledger and returned rows inherit FlowGraph's own 512-edge maximum:
    # a full 512-edge network is admitted and accounted exactly.
    full = compute_multicommodity_flow_profile(many_edge_flow(64, 512))
    assert len(full.edge_profiles) == 512
    assert full.work.logical_steps_per_call == 2 * (512 + 1 + 512 + 64 + 3 * 512)

    with pytest.raises(ValidationError, match=r"at most 512 item"):
        many_edge_flow(64, 513)


def test_cell_budgets_admit_commodities_without_a_fixed_ceiling() -> None:
    # Commodity count is bounded by the derived commodity-vertex and
    # commodity-edge cell budgets rather than an independent fixed ceiling:
    # 17 commodities over two vertices occupy only 34 cells of constant size.
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
    assert unrouted.work.logical_steps_per_call == 2 * (1 + 17 + 1 + 34 + 3)

    # Each commodity occupies at least two distinct commodity-vertex cells,
    # so exactly half of the 512-cell divergence budget, 256 commodities, is
    # admitted over two vertices; one more exceeds the dense table.
    full = compute_multicommodity_flow_profile(dense_commodities(256))
    assert len(full.divergences) == MAX_COMMODITY_VERTEX_CELLS
    assert full.work.rational_negations_per_pass == 256
    assert full.work.logical_steps_per_call == 2 * (1 + 256 + 1 + 512 + 3)

    with pytest.raises(ValidationError, match="commodity-by-vertex"):
        dense_commodities(257)


def test_result_envelope_rejects_wide_tensors_with_large_operands() -> None:
    from itertools import combinations

    def comb_edge_tensor(capacity: CanonicalRational) -> MulticommodityFlow:
        # 128 edges keep the echoed source under the 10 MiB canonical encoder
        # while the priced derived rows cross the 8 MiB result envelope.
        edges = tuple(
            CapacitatedEdge(source=source, target=target, capacity=capacity)
            for source, target in combinations(range(32), 2)
        )[:128]
        return MulticommodityFlow(
            network=FlowGraph(vertex_count=32, edges=edges),
            commodities=tuple(
                CommodityDemand(
                    commodity_id=f"c{index:02d}", source=0, sink=31, demand=q(1)
                )
                for index in range(4)
            ),
        )

    # Every edge slack is priced at its own worst-case digit bound, so 128
    # 30,000-digit capacities price the edge rows alone above the aggregate
    # envelope and the tensor is rejected before any backend runs.
    with pytest.raises(ValidationError, match="aggregate result bound"):
        comb_edge_tensor(CanonicalRational(num="9" * 30_000, den="1"))

    # Halving the operand size keeps every priced row inside the envelope:
    # large exact operands are admitted whenever the components they can
    # reach stay bounded, never because of a fixed input cap.
    admitted = comb_edge_tensor(CanonicalRational(num="9" * 5_000, den="1"))
    result = compute_multicommodity_flow_profile(admitted)
    assert result.capacity_feasible is True
    assert result.work.edge_cells == len(result.flow.network.edges)


def test_result_replay_rejects_forged_source_and_derived_ledger_fields() -> None:
    result = compute_multicommodity_flow_profile(shared_bottleneck_flow())
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
