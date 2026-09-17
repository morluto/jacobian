"""Exact multicommodity-flow solves: feasibility, congestion, unsplittable routing."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.graphs.flows._models import CapacitatedEdge, FlowGraph
from jacobian.math.graphs.flows.multicommodity._models import (
    CommodityDemand,
    MinimumCongestionRequest,
    MinimumCongestionResult,
    MulticommodityFeasibilityRequest,
    MulticommodityFeasibilityResult,
    UnsplittablePath,
    UnsplittableRouting,
    UnsplittableRoutingCheckRequest,
    UnsplittableRoutingFindRequest,
    UnsplittableRoutingFindResult,
)
from jacobian.math.graphs.flows.multicommodity._tools import (
    TOOLS,
    _run_unsplittable_routing_check,
    _run_unsplittable_routing_find,
)
from jacobian.math.graphs.flows.multicommodity.operations import (
    check_multicommodity_flow_witness,
    check_unsplittable_routing,
    find_unsplittable_routing,
    solve_minimum_congestion,
    solve_multicommodity_feasibility,
    verify_minimum_congestion,
    verify_multicommodity_feasibility,
    verify_unsplittable_routing_find,
)


def _tool(operation_id: str) -> MathTool[Any, Any]:
    return next(tool for tool in TOOLS if tool.operation_id == operation_id)


def q(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def _bottleneck() -> FlowGraph:
    return FlowGraph(
        vertex_count=4,
        edges=(
            CapacitatedEdge(source=0, target=2, capacity=q(2)),
            CapacitatedEdge(source=1, target=2, capacity=q(2)),
            CapacitatedEdge(source=2, target=3, capacity=q(3)),
        ),
    )


def _demands() -> tuple[CommodityDemand, ...]:
    return (
        CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(1)),
        CommodityDemand(commodity_id="b", source=1, sink=3, demand=q(2)),
    )


def _unit_demands() -> tuple[CommodityDemand, ...]:
    return (
        CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(1)),
        CommodityDemand(commodity_id="b", source=1, sink=3, demand=q(1)),
    )


def _diamond() -> FlowGraph:
    return FlowGraph(
        vertex_count=4,
        edges=(
            CapacitatedEdge(source=0, target=1, capacity=q(1)),
            CapacitatedEdge(source=0, target=2, capacity=q(1)),
            CapacitatedEdge(source=1, target=3, capacity=q(1)),
            CapacitatedEdge(source=2, target=3, capacity=q(1)),
        ),
    )


def _replay_farkas(
    network: FlowGraph,
    commodities: tuple[CommodityDemand, ...],
    potentials: tuple[tuple[Fraction, ...], ...],
    prices: tuple[Fraction, ...],
    value: Fraction,
) -> None:
    """Independently replay a network Farkas certificate in the tests."""

    assert len(potentials) == len(commodities)
    assert all(len(row) == network.vertex_count for row in potentials)
    assert len(prices) == len(network.edges)
    assert all(price >= 0 for price in prices)
    for commodity_index in range(len(commodities)):
        for edge_index, edge in enumerate(network.edges):
            slack = (
                prices[edge_index]
                + potentials[commodity_index][edge.source]
                - potentials[commodity_index][edge.target]
            )
            assert slack >= 0
    total = sum(
        commodity.demand.as_fraction()
        * (
            potentials[commodity_index][commodity.source]
            - potentials[commodity_index][commodity.sink]
        )
        for commodity_index, commodity in enumerate(commodities)
    ) + sum(
        price * edge.capacity.as_fraction()
        for price, edge in zip(prices, network.edges, strict=True)
    )
    assert total == value
    assert value < 0


class TestCatalogRegistry:
    def test_catalog_contains_the_solve_family(self) -> None:
        assert {tool.operation_id for tool in TOOLS} == {
            "network.multicommodity_flow.profile.compute",
            "network.multicommodity_flow.witness.check",
            "network.multicommodity_flow.feasibility.compute",
            "network.multicommodity_flow.minimum_congestion.compute",
            "network.multicommodity_flow.unsplittable_routing.check",
            "network.multicommodity_flow.unsplittable_routing.find",
        }


class TestFeasibility:
    def test_bottleneck_feasible_with_exact_tensor(self) -> None:
        result = solve_multicommodity_feasibility(_bottleneck(), _demands())

        assert result.status == "FEASIBLE"
        assert result.flow is not None
        assert result.certificate is None
        assert check_multicommodity_flow_witness(result.flow).status == "FEASIBLE"
        assert verify_multicommodity_feasibility(result)

    def test_over_demand_is_infeasible_with_farkas(self) -> None:
        commodities = (
            CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(2)),
            CommodityDemand(commodity_id="b", source=1, sink=3, demand=q(2)),
        )
        result = solve_multicommodity_feasibility(_bottleneck(), commodities)

        assert result.status == "INFEASIBLE"
        assert result.flow is None
        assert result.certificate is not None
        _replay_farkas(
            _bottleneck(),
            commodities,
            tuple(
                tuple(value.as_fraction() for value in row)
                for row in result.certificate.node_potentials
            ),
            tuple(price.as_fraction() for price in result.certificate.edge_prices),
            result.certificate.farkas_value.as_fraction(),
        )
        assert verify_multicommodity_feasibility(result)

    def test_disconnected_commodity_is_infeasible(self) -> None:
        network = FlowGraph(
            vertex_count=3,
            edges=(CapacitatedEdge(source=0, target=1, capacity=q(5)),),
        )
        commodities = (
            CommodityDemand(commodity_id="a", source=0, sink=2, demand=q(1)),
        )
        result = solve_multicommodity_feasibility(network, commodities)

        assert result.status == "INFEASIBLE"
        assert result.certificate is not None
        _replay_farkas(
            network,
            commodities,
            tuple(
                tuple(value.as_fraction() for value in row)
                for row in result.certificate.node_potentials
            ),
            tuple(price.as_fraction() for price in result.certificate.edge_prices),
            result.certificate.farkas_value.as_fraction(),
        )

    def test_search_is_deterministic(self) -> None:
        first = solve_multicommodity_feasibility(_bottleneck(), _demands())
        second = solve_multicommodity_feasibility(_bottleneck(), _demands())

        assert first == second

    def test_native_and_catalog_paths_agree(self) -> None:
        tool = _tool("network.multicommodity_flow.feasibility.compute")
        request = MulticommodityFeasibilityRequest(
            network=_bottleneck(), commodities=_demands()
        )

        assert tool.run(request) == solve_multicommodity_feasibility(
            _bottleneck(), _demands()
        )

    def test_round_trip_and_forgery(self) -> None:
        result = solve_multicommodity_feasibility(_bottleneck(), _demands())
        restored = MulticommodityFeasibilityResult.model_validate_json(
            result.model_dump_json()
        )

        assert restored == result
        assert verify_multicommodity_feasibility(restored)
        forged = json.loads(restored.model_dump_json())
        forged["status"] = "INFEASIBLE"
        forged["flow"] = None
        with pytest.raises(ValidationError):
            MulticommodityFeasibilityResult.model_validate_json(json.dumps(forged))

    def test_unsorted_contract_is_rejected(self) -> None:
        network = FlowGraph(
            vertex_count=4,
            edges=(
                CapacitatedEdge(source=1, target=2, capacity=q(2)),
                CapacitatedEdge(source=0, target=2, capacity=q(2)),
                CapacitatedEdge(source=2, target=3, capacity=q(3)),
            ),
        )
        with pytest.raises(OperationDomainValidationError):
            solve_multicommodity_feasibility(network, _demands())

    def test_oversized_program_is_a_resource_boundary(self) -> None:
        network = FlowGraph(
            vertex_count=8,
            edges=tuple(
                CapacitatedEdge(source=index, target=index + 1, capacity=q(1))
                for index in range(7)
            ),
        )
        commodities = tuple(
            CommodityDemand(commodity_id=f"c{index}", source=0, sink=7, demand=q(1))
            for index in range(5)
        )
        # 5 commodities x 7 edges = 35 variables exceeds the 32-variable LP source.
        from jacobian.catalog.models import OperationResourceAdmissionError

        with pytest.raises(OperationResourceAdmissionError) as exc_info:
            solve_multicommodity_feasibility(network, commodities)
        assert "envelope" in exc_info.value.errors()[0]["type"]


class TestMinimumCongestion:
    def test_bottleneck_minimum_is_one(self) -> None:
        result = solve_minimum_congestion(_bottleneck(), _demands())

        assert result.status == "FEASIBLE"
        assert result.congestion is not None
        assert result.congestion.as_fraction() == 1
        assert result.flow is not None
        assert verify_minimum_congestion(result)

    def test_fractional_minimum_congestion(self) -> None:
        network = FlowGraph(
            vertex_count=2,
            edges=(CapacitatedEdge(source=0, target=1, capacity=q(2)),),
        )
        commodities = (
            CommodityDemand(commodity_id="a", source=0, sink=1, demand=q(1)),
        )
        result = solve_minimum_congestion(network, commodities)

        assert result.status == "FEASIBLE"
        assert result.congestion is not None
        assert result.congestion.as_fraction() == Fraction(1, 2)

    def test_single_commodity_matches_max_flow_ratio(self) -> None:
        # Independent cross-check: min congestion = demand / max flow value.
        from jacobian.math.graphs.flows.operations import max_flow

        network = FlowGraph(
            vertex_count=4,
            edges=(
                CapacitatedEdge(source=0, target=1, capacity=q(3)),
                CapacitatedEdge(source=0, target=2, capacity=q(2)),
                CapacitatedEdge(source=1, target=3, capacity=q(2)),
                CapacitatedEdge(source=2, target=3, capacity=q(2)),
            ),
        )
        commodities = (
            CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(3)),
        )
        result = solve_minimum_congestion(network, commodities)
        flow_value, _ = max_flow(network, 0, 3)

        assert result.status == "FEASIBLE"
        assert result.congestion is not None
        assert (
            result.congestion.as_fraction() == Fraction(3, 1) / flow_value.as_fraction()
        )
        assert flow_value.as_fraction() == 4

    def test_zero_capacity_disconnect_is_infeasible(self) -> None:
        network = FlowGraph(
            vertex_count=4,
            edges=(
                CapacitatedEdge(source=0, target=2, capacity=q(2)),
                CapacitatedEdge(source=1, target=2, capacity=q(0)),
                CapacitatedEdge(source=2, target=3, capacity=q(3)),
            ),
        )
        result = solve_minimum_congestion(network, _unit_demands())

        assert result.status == "INFEASIBLE"
        assert result.certificate is not None
        _replay_farkas(
            network,
            _unit_demands(),
            tuple(
                tuple(value.as_fraction() for value in row)
                for row in result.certificate.node_potentials
            ),
            tuple(price.as_fraction() for price in result.certificate.edge_prices),
            result.certificate.farkas_value.as_fraction(),
        )
        assert verify_minimum_congestion(result)

    def test_native_and_catalog_paths_agree(self) -> None:
        tool = _tool("network.multicommodity_flow.minimum_congestion.compute")
        request = MinimumCongestionRequest(
            network=_bottleneck(), commodities=_demands()
        )

        assert tool.run(request) == solve_minimum_congestion(_bottleneck(), _demands())

    def test_round_trip_and_forgery(self) -> None:
        result = solve_minimum_congestion(_bottleneck(), _demands())
        restored = MinimumCongestionResult.model_validate_json(result.model_dump_json())

        assert restored == result
        assert verify_minimum_congestion(restored)
        forged = json.loads(restored.model_dump_json())
        forged["congestion"] = {"num": "2", "den": "1"}
        forged_claim = MinimumCongestionResult.model_validate_json(json.dumps(forged))
        assert not verify_minimum_congestion(forged_claim)


class TestUnsplittableCheck:
    def test_feasible_routing(self) -> None:
        routing = UnsplittableRouting(
            network=_bottleneck(),
            commodities=_unit_demands(),
            paths=(
                UnsplittablePath(commodity_id="a", vertices=(0, 2, 3)),
                UnsplittablePath(commodity_id="b", vertices=(1, 2, 3)),
            ),
        )
        result = check_unsplittable_routing(routing)

        assert result.status == "FEASIBLE"
        assert result.path_violations == ()
        assert result.edge_violations == ()

    def test_non_edge_step_is_infeasible(self) -> None:
        routing = UnsplittableRouting(
            network=_bottleneck(),
            commodities=_unit_demands(),
            paths=(
                UnsplittablePath(commodity_id="a", vertices=(0, 2, 3)),
                UnsplittablePath(commodity_id="b", vertices=(1, 0, 2, 3)),
            ),
        )
        result = check_unsplittable_routing(routing)

        assert result.status == "INFEASIBLE"
        assert [row.commodity_id for row in result.path_violations] == ["b"]

    def test_endpoint_mismatch_is_infeasible(self) -> None:
        routing = UnsplittableRouting(
            network=_bottleneck(),
            commodities=_unit_demands(),
            paths=(
                UnsplittablePath(commodity_id="a", vertices=(0, 2, 3)),
                UnsplittablePath(commodity_id="b", vertices=(1, 2, 0)),
            ),
        )
        result = check_unsplittable_routing(routing)

        assert result.status == "INFEASIBLE"
        assert [row.commodity_id for row in result.path_violations] == ["b"]

    def test_capacity_overload_is_infeasible(self) -> None:
        commodities = (
            CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(2)),
            CommodityDemand(commodity_id="b", source=1, sink=3, demand=q(2)),
        )
        routing = UnsplittableRouting(
            network=_bottleneck(),
            commodities=commodities,
            paths=(
                UnsplittablePath(commodity_id="a", vertices=(0, 2, 3)),
                UnsplittablePath(commodity_id="b", vertices=(1, 2, 3)),
            ),
        )
        result = check_unsplittable_routing(routing)

        assert result.status == "INFEASIBLE"
        assert [(row.source, row.target) for row in result.edge_violations] == [(2, 3)]

    def test_repeated_vertex_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            UnsplittablePath(commodity_id="a", vertices=(0, 2, 0, 3))

    def test_duplicate_paths_per_commodity_are_rejected(self) -> None:
        # The checker adds one demand per path, so a routing with two paths for
        # one commodity would silently double its load.
        commodities = (
            CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(1)),
        )

        with pytest.raises(ValidationError):
            UnsplittableRouting(
                network=_bottleneck(),
                commodities=commodities,
                paths=(
                    UnsplittablePath(commodity_id="a", vertices=(0, 2, 3)),
                    UnsplittablePath(commodity_id="a", vertices=(0, 2, 3)),
                ),
            )

    def test_native_and_catalog_paths_agree(self) -> None:
        routing = UnsplittableRouting(
            network=_bottleneck(),
            commodities=_unit_demands(),
            paths=(
                UnsplittablePath(commodity_id="a", vertices=(0, 2, 3)),
                UnsplittablePath(commodity_id="b", vertices=(1, 2, 3)),
            ),
        )
        assert _run_unsplittable_routing_check(
            UnsplittableRoutingCheckRequest(routing=routing)
        ) == check_unsplittable_routing(routing)


class TestUnsplittableFind:
    def test_bottleneck_routing_is_found(self) -> None:
        result = find_unsplittable_routing(_bottleneck(), _unit_demands(), 256, 50000)

        assert result.status == "FOUND"
        assert result.routing is not None
        assert result.certificate is not None
        assert result.certificate.status == "FEASIBLE"
        assert verify_unsplittable_routing_find(result)

    def test_overload_exhausts(self) -> None:
        commodities = (
            CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(2)),
            CommodityDemand(commodity_id="b", source=1, sink=3, demand=q(2)),
        )
        result = find_unsplittable_routing(_bottleneck(), commodities, 256, 50000)

        assert result.status == "EXHAUSTED"
        assert result.combinations_examined == result.total_combinations == 1
        assert result.paths_per_commodity == (1, 1)

    def test_truncated_search_reports_unknown(self) -> None:
        commodities = (
            CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(1)),
            CommodityDemand(commodity_id="b", source=0, sink=3, demand=q(1)),
        )
        result = find_unsplittable_routing(_diamond(), commodities, 256, 1)

        assert result.status == "UNKNOWN"
        assert result.stop_reason == "COMBINATION_BUDGET_EXCEEDED"
        assert result.combinations_examined == 1
        assert result.total_combinations == 4

    def test_path_cap_reports_unknown(self) -> None:
        commodities = (
            CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(1)),
        )
        result = find_unsplittable_routing(_diamond(), commodities, 1, 50000)

        assert result.status == "UNKNOWN"
        assert result.stop_reason == "PATH_CAP_EXCEEDED"
        assert result.current_commodity == "a"

    def test_diamond_exhausts_when_capacities_block(self) -> None:
        network = FlowGraph(
            vertex_count=4,
            edges=(
                CapacitatedEdge(source=0, target=1, capacity=q(1)),
                CapacitatedEdge(source=0, target=2, capacity=q(1)),
                CapacitatedEdge(source=1, target=3, capacity=q(0)),
                CapacitatedEdge(source=2, target=3, capacity=q(0)),
            ),
        )
        commodities = (
            CommodityDemand(commodity_id="a", source=0, sink=3, demand=q(1)),
        )
        result = find_unsplittable_routing(network, commodities, 256, 50000)

        assert result.status == "EXHAUSTED"
        assert result.paths_per_commodity == (2,)
        assert result.combinations_examined == result.total_combinations == 2

    def test_native_and_catalog_paths_agree(self) -> None:
        request = UnsplittableRoutingFindRequest(
            network=_bottleneck(),
            commodities=_unit_demands(),
            max_paths_per_commodity=256,
            combination_budget=50000,
        )
        assert _run_unsplittable_routing_find(request) == find_unsplittable_routing(
            _bottleneck(), _unit_demands(), 256, 50000
        )

    def test_round_trip_and_forgery(self) -> None:
        result = find_unsplittable_routing(_bottleneck(), _unit_demands(), 256, 50000)
        restored = UnsplittableRoutingFindResult.model_validate_json(
            result.model_dump_json()
        )

        assert restored == result
        assert verify_unsplittable_routing_find(restored)
        forged = json.loads(restored.model_dump_json())
        forged["combinations_examined"] = 0
        forged_claim = UnsplittableRoutingFindResult.model_validate_json(
            json.dumps(forged)
        )
        assert not verify_unsplittable_routing_find(forged_claim)

    def test_budgets_are_domain_boundaries(self) -> None:
        with pytest.raises(OperationDomainValidationError):
            find_unsplittable_routing(_bottleneck(), _unit_demands(), 256, 0)
        with pytest.raises(OperationDomainValidationError):
            find_unsplittable_routing(_bottleneck(), _unit_demands(), 0, 50000)
