"""Exact multicommodity-flow operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.flows.multicommodity._models import (
    MAX_DECOMPOSITION_INTERMEDIATE_DIGITS,
    MAX_DECOMPOSITION_TRAVERSAL_STEPS,
    MAX_DECOMPOSITION_VERTEX_CELLS,
    MinimumCongestionRequest,
    MinimumCongestionResult,
    MulticommodityFeasibilityRequest,
    MulticommodityFeasibilityResult,
    MulticommodityFlowDecompositionRequest,
    MulticommodityFlowDecompositionResult,
    MulticommodityFlowProfileRequest,
    MulticommodityFlowProfileResult,
    MulticommodityFlowWitnessCheckRequest,
    MulticommodityFlowWitnessCheckResult,
    UnsplittableRoutingCheckRequest,
    UnsplittableRoutingCheckResult,
    UnsplittableRoutingFindRequest,
    UnsplittableRoutingFindResult,
)
from jacobian.math.graphs.flows.multicommodity.operations import (
    check_multicommodity_flow_witness,
    check_unsplittable_routing,
    compute_multicommodity_flow_profile,
    decompose_multicommodity_flow,
    find_unsplittable_routing,
    solve_minimum_congestion,
    solve_multicommodity_feasibility,
)


def _run_multicommodity_flow_profile(
    request: MulticommodityFlowProfileRequest,
) -> MulticommodityFlowProfileResult:
    return compute_multicommodity_flow_profile(request.flow)


def _run_multicommodity_flow_witness_check(
    request: MulticommodityFlowWitnessCheckRequest,
) -> MulticommodityFlowWitnessCheckResult:
    return check_multicommodity_flow_witness(request.flow)


def _run_multicommodity_flow_decomposition(
    request: MulticommodityFlowDecompositionRequest,
) -> MulticommodityFlowDecompositionResult:
    return decompose_multicommodity_flow(request.flow)


def _run_multicommodity_feasibility(
    request: MulticommodityFeasibilityRequest,
) -> MulticommodityFeasibilityResult:
    return solve_multicommodity_feasibility(request.network, request.commodities)


def _run_minimum_congestion(
    request: MinimumCongestionRequest,
) -> MinimumCongestionResult:
    return solve_minimum_congestion(request.network, request.commodities)


def _run_unsplittable_routing_check(
    request: UnsplittableRoutingCheckRequest,
) -> UnsplittableRoutingCheckResult:
    return check_unsplittable_routing(request.routing)


def _run_unsplittable_routing_find(
    request: UnsplittableRoutingFindRequest,
) -> UnsplittableRoutingFindResult:
    return find_unsplittable_routing(
        request.network,
        request.commodities,
        request.max_paths_per_commodity,
        request.combination_budget,
    )


_BOTTLENECK_NETWORK: dict[str, object] = {
    "vertex_count": 4,
    "edges": [
        {"source": 0, "target": 2, "capacity": {"num": "2", "den": "1"}},
        {"source": 1, "target": 2, "capacity": {"num": "2", "den": "1"}},
        {"source": 2, "target": 3, "capacity": {"num": "3", "den": "1"}},
    ],
}


_BOTTLENECK_COMMODITIES: list[object] = [
    {"commodity_id": "a", "source": 0, "sink": 3, "demand": {"num": "1", "den": "1"}},
    {"commodity_id": "b", "source": 1, "sink": 3, "demand": {"num": "2", "den": "1"}},
]


_BOTTLENECK_UNIT_COMMODITIES: list[object] = [
    {"commodity_id": "a", "source": 0, "sink": 3, "demand": {"num": "1", "den": "1"}},
    {"commodity_id": "b", "source": 1, "sink": 3, "demand": {"num": "1", "den": "1"}},
]


TOOLS: MathTools = (
    MathTool(
        operation_id="network.multicommodity_flow.profile.compute",
        title="Compute an exact multicommodity-flow profile",
        description=(
            "Compute exact commodity conservation, aggregate edge loads, signed "
            "capacity slacks, capacity feasibility, and congestion for one "
            "canonical sparse rational multicommodity-flow tensor. The result "
            "retains the complete network, demands, and tensor; it "
            "does not search for a flow or solve an optimization problem."
        ),
        request_type=MulticommodityFlowProfileRequest,
        result_type=MulticommodityFlowProfileResult,
        run=_run_multicommodity_flow_profile,
        tags=(
            "network",
            "multicommodity-flow",
            "flow-profile",
            "conservation",
            "capacity",
            "exact",
            "bounded",
        ),
        examples=(
            OperationExample(
                name="two_commodities_share_a_bottleneck",
                description="Profile two exact commodity flows sharing a directed bottleneck; "
                "network edges, commodities, and nonzero entries must use their "
                "published canonical sort orders.",
                input={
                    "flow": {
                        "network": {
                            "vertex_count": 4,
                            "edges": [
                                {
                                    "source": 0,
                                    "target": 2,
                                    "capacity": {"num": "2", "den": "1"},
                                },
                                {
                                    "source": 1,
                                    "target": 2,
                                    "capacity": {"num": "2", "den": "1"},
                                },
                                {
                                    "source": 2,
                                    "target": 3,
                                    "capacity": {"num": "3", "den": "1"},
                                },
                            ],
                        },
                        "commodities": [
                            {
                                "commodity_id": "a",
                                "source": 0,
                                "sink": 3,
                                "demand": {"num": "1", "den": "1"},
                            },
                            {
                                "commodity_id": "b",
                                "source": 1,
                                "sink": 3,
                                "demand": {"num": "2", "den": "1"},
                            },
                        ],
                        "entries": [
                            {
                                "commodity_id": "a",
                                "source": 0,
                                "target": 2,
                                "amount": {"num": "1", "den": "1"},
                            },
                            {
                                "commodity_id": "a",
                                "source": 2,
                                "target": 3,
                                "amount": {"num": "1", "den": "1"},
                            },
                            {
                                "commodity_id": "b",
                                "source": 1,
                                "target": 2,
                                "amount": {"num": "2", "den": "1"},
                            },
                            {
                                "commodity_id": "b",
                                "source": 2,
                                "target": 3,
                                "amount": {"num": "2", "den": "1"},
                            },
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="network.multicommodity_flow.witness.check",
        title="Check an exact multicommodity-flow witness",
        description=(
            "Replay one submitted canonical sparse commodity-by-edge flow in "
            "exact rational arithmetic. For every commodity and vertex recompute "
            "outgoing minus incoming flow and compare it with the declared "
            "terminals and demand (source +d, sink -d, otherwise 0), and for "
            "every directed edge compare the aggregate load with its capacity. "
            "Return FEASIBLE with the divergence ledger, per-edge loads and "
            "slacks, congestion, and work, or INFEASIBLE with the exact "
            "violating commodities, commodity-vertex cells, and edges. Entries "
            "are nonnegative by construction; omitted entries are exact zero. "
            "No floating-point solver participates."
        ),
        request_type=MulticommodityFlowWitnessCheckRequest,
        result_type=MulticommodityFlowWitnessCheckResult,
        run=_run_multicommodity_flow_witness_check,
        tags=(
            "network",
            "multicommodity-flow",
            "witness",
            "feasibility",
            "conservation",
            "capacity",
            "exact",
            "bounded",
        ),
        examples=(
            OperationExample(
                name="two_commodities_share_a_bottleneck",
                description="Check two exact commodity flows sharing a directed bottleneck; "
                "network edges, commodities, and nonzero entries must use their "
                "published canonical sort orders.",
                input={
                    "flow": {
                        "network": {
                            "vertex_count": 4,
                            "edges": [
                                {
                                    "source": 0,
                                    "target": 2,
                                    "capacity": {"num": "2", "den": "1"},
                                },
                                {
                                    "source": 1,
                                    "target": 2,
                                    "capacity": {"num": "2", "den": "1"},
                                },
                                {
                                    "source": 2,
                                    "target": 3,
                                    "capacity": {"num": "3", "den": "1"},
                                },
                            ],
                        },
                        "commodities": [
                            {
                                "commodity_id": "a",
                                "source": 0,
                                "sink": 3,
                                "demand": {"num": "1", "den": "1"},
                            },
                            {
                                "commodity_id": "b",
                                "source": 1,
                                "sink": 3,
                                "demand": {"num": "2", "den": "1"},
                            },
                        ],
                        "entries": [
                            {
                                "commodity_id": "a",
                                "source": 0,
                                "target": 2,
                                "amount": {"num": "1", "den": "1"},
                            },
                            {
                                "commodity_id": "a",
                                "source": 2,
                                "target": 3,
                                "amount": {"num": "1", "den": "1"},
                            },
                            {
                                "commodity_id": "b",
                                "source": 1,
                                "target": 2,
                                "amount": {"num": "2", "den": "1"},
                            },
                            {
                                "commodity_id": "b",
                                "source": 2,
                                "target": 3,
                                "amount": {"num": "2", "den": "1"},
                            },
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="network.multicommodity_flow.decomposition.compute",
        title="Decompose an exact multicommodity flow into paths and cycles",
        description=(
            "Decompose each nonnegative commodity-edge tensor into deterministic "
            "simple directed path and cycle terms. The terms retain the source "
            "network and commodities and reconstruct every tensor entry exactly; "
            "circulation is returned as cycles and empty support as empty tuples. "
            "Admission requires nonzero_entries * (vertex_count + 1) <= "
            f"{MAX_DECOMPOSITION_VERTEX_CELLS} output vertex cells, bounds "
            f"residual rational intermediates to {MAX_DECOMPOSITION_INTERMEDIATE_DIGITS} "
            f"decimal digits, and admits residual traversal within the "
            f"{MAX_DECOMPOSITION_TRAVERSAL_STEPS}-step envelope before any "
            "decomposition arithmetic."
        ),
        request_type=MulticommodityFlowDecompositionRequest,
        result_type=MulticommodityFlowDecompositionResult,
        run=_run_multicommodity_flow_decomposition,
        tags=(
            "network",
            "multicommodity-flow",
            "decomposition",
            "paths",
            "cycles",
            "exact",
            "bounded",
        ),
        discovery_terms=(
            "multicommodity flow decomposition",
            "path cycle decomposition",
            "flow tensor reconstruction",
        ),
        examples=(
            OperationExample(
                name="path_and_circulation",
                description=(
                    "Decompose one exact path plus a directed circulation; the "
                    "flow tensor is nonnegative and all network edges and entries "
                    "use canonical source order."
                ),
                input={
                    "flow": {
                        "network": {
                            "vertex_count": 4,
                            "edges": [
                                {
                                    "source": 0,
                                    "target": 1,
                                    "capacity": {"num": "2", "den": "1"},
                                },
                                {
                                    "source": 1,
                                    "target": 2,
                                    "capacity": {"num": "2", "den": "1"},
                                },
                                {
                                    "source": 1,
                                    "target": 3,
                                    "capacity": {"num": "2", "den": "1"},
                                },
                                {
                                    "source": 2,
                                    "target": 1,
                                    "capacity": {"num": "2", "den": "1"},
                                },
                            ],
                        },
                        "commodities": [
                            {
                                "commodity_id": "a",
                                "source": 0,
                                "sink": 3,
                                "demand": {"num": "1", "den": "1"},
                            }
                        ],
                        "entries": [
                            {
                                "commodity_id": "a",
                                "source": 0,
                                "target": 1,
                                "amount": {"num": "1", "den": "1"},
                            },
                            {
                                "commodity_id": "a",
                                "source": 1,
                                "target": 2,
                                "amount": {"num": "1", "den": "2"},
                            },
                            {
                                "commodity_id": "a",
                                "source": 1,
                                "target": 3,
                                "amount": {"num": "1", "den": "1"},
                            },
                            {
                                "commodity_id": "a",
                                "source": 2,
                                "target": 1,
                                "amount": {"num": "1", "den": "2"},
                            },
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="network.multicommodity_flow.feasibility.compute",
        title="Solve exact multicommodity-flow feasibility",
        description=(
            "Solve rational multicommodity-flow feasibility on the exact "
            "rational LP backend; no floating-point solve participates. "
            "FEASIBLE carries an exact commodity-by-edge tensor verified by "
            "the witness checker; INFEASIBLE carries a network Farkas "
            "certificate (node potentials and edge prices) whose balance "
            "replays in exact arithmetic; UNKNOWN carries the bounded "
            "backend reason. A truncated computation never becomes a verdict."
        ),
        request_type=MulticommodityFeasibilityRequest,
        result_type=MulticommodityFeasibilityResult,
        run=_run_multicommodity_feasibility,
        tags=(
            "network",
            "multicommodity-flow",
            "feasibility",
            "exact",
            "bounded",
        ),
        discovery_terms=(
            "multicommodity flow feasibility",
            "exact rational flow",
            "Farkas certificate",
        ),
        examples=(
            OperationExample(
                name="bottleneck_feasible_demands",
                description="Route demands 1 and 2 through the shared "
                "capacity-3 bottleneck; the exact LP backend returns a "
                "verified rational tensor.",
                input={
                    "network": _BOTTLENECK_NETWORK,
                    "commodities": _BOTTLENECK_COMMODITIES,
                },
            ),
            OperationExample(
                name="bottleneck_over_demand_infeasible",
                description="Demands 2 and 2 exceed the shared capacity-3 "
                "bottleneck; the backend returns a Farkas certificate.",
                input={
                    "network": _BOTTLENECK_NETWORK,
                    "commodities": [
                        {
                            "commodity_id": "a",
                            "source": 0,
                            "sink": 3,
                            "demand": {"num": "2", "den": "1"},
                        },
                        {
                            "commodity_id": "b",
                            "source": 1,
                            "sink": 3,
                            "demand": {"num": "2", "den": "1"},
                        },
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="network.multicommodity_flow.minimum_congestion.compute",
        title="Minimize exact multicommodity congestion",
        description=(
            "Minimize the exact congestion ratio over rational multicommodity "
            "flows on the exact rational LP backend. FEASIBLE carries the "
            "exact minimum ratio with an attaining tensor; INFEASIBLE carries "
            "a network Farkas certificate for the underlying routability; "
            "UNKNOWN carries the bounded backend reason."
        ),
        request_type=MinimumCongestionRequest,
        result_type=MinimumCongestionResult,
        run=_run_minimum_congestion,
        tags=(
            "network",
            "multicommodity-flow",
            "congestion",
            "exact",
            "bounded",
        ),
        discovery_terms=(
            "minimum congestion",
            "multicommodity flow optimum",
            "exact rational flow",
        ),
        examples=(
            OperationExample(
                name="bottleneck_minimum_congestion_one",
                description="The shared bottleneck saturates exactly: the "
                "minimum congestion ratio is 1 with an attaining tensor.",
                input={
                    "network": _BOTTLENECK_NETWORK,
                    "commodities": _BOTTLENECK_COMMODITIES,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="network.multicommodity_flow.unsplittable_routing.check",
        title="Check one-path-per-commodity routing",
        description=(
            "Check one submitted simple source-to-sink path per commodity: "
            "every path step must be a network edge, endpoints must match "
            "the commodity terminals, and aggregate per-edge loads (one "
            "demand per using commodity) must fit capacities. FEASIBLE "
            "carries the congestion; INFEASIBLE names every defect."
        ),
        request_type=UnsplittableRoutingCheckRequest,
        result_type=UnsplittableRoutingCheckResult,
        run=_run_unsplittable_routing_check,
        tags=(
            "network",
            "unsplittable-routing",
            "exact",
            "bounded",
        ),
        discovery_terms=(
            "unsplittable flow",
            "single path routing",
            "one path per commodity",
        ),
        examples=(
            OperationExample(
                name="bottleneck_unsplittable_routing",
                description="Route each unit demand along its unique "
                "bottleneck path; aggregate loads fit exactly.",
                input={
                    "routing": {
                        "network": _BOTTLENECK_NETWORK,
                        "commodities": _BOTTLENECK_UNIT_COMMODITIES,
                        "paths": [
                            {"commodity_id": "a", "vertices": [0, 2, 3]},
                            {"commodity_id": "b", "vertices": [1, 2, 3]},
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="network.multicommodity_flow.unsplittable_routing.find",
        title="Find one-path-per-commodity routing by bounded search",
        description=(
            "Enumerate simple source-to-sink paths per commodity up to a cap, "
            "then path combinations up to a budget, deciding each with the "
            "exact routing checker. FOUND carries the first feasible routing "
            "with its certificate; EXHAUSTED carries the exact combination "
            "receipt; UNKNOWN carries the bounded stop reason. A truncated "
            "search never yields a negative conclusion."
        ),
        request_type=UnsplittableRoutingFindRequest,
        result_type=UnsplittableRoutingFindResult,
        run=_run_unsplittable_routing_find,
        tags=(
            "network",
            "unsplittable-routing",
            "search",
            "exact",
            "bounded",
        ),
        discovery_terms=(
            "unsplittable flow search",
            "single path routing",
            "bounded path enumeration",
        ),
        examples=(
            OperationExample(
                name="bottleneck_routing_search",
                description="Find the unique feasible one-path routing "
                "through the shared bottleneck.",
                input={
                    "network": _BOTTLENECK_NETWORK,
                    "commodities": _BOTTLENECK_UNIT_COMMODITIES,
                    "max_paths_per_commodity": 256,
                    "combination_budget": 50000,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
