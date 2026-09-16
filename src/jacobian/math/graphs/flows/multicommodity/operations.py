"""Public operation adapters for exact multicommodity-flow profiles."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.graphs.flows.multicommodity._kernel import profile_components
from jacobian.math.graphs.flows.multicommodity._models import (
    AdmittedProfileScan,
    CommodityVertexViolation,
    EdgeCapacityViolation,
    MulticommodityFlow,
    MulticommodityFlowProfileResult,
    MulticommodityFlowWitnessCheckResult,
    _require_profile_output_admission,
)


def compute_multicommodity_flow_profile(
    flow: MulticommodityFlow,
) -> MulticommodityFlowProfileResult:
    """Compute the exact bounded load and conservation profile of one tensor.

    The canonical tensor value carries only representation bounds; this
    execution boundary performs the profile's semantic admission and exact
    computation. Request parsing remains structural.
    """

    return _profile_result(flow, _require_profile_output_admission(flow))


def _profile_result(
    flow: MulticommodityFlow,
    admitted: AdmittedProfileScan,
) -> MulticommodityFlowProfileResult:
    (
        divergences,
        edge_profiles,
        all_demands_routed,
        capacity_feasible,
        congestion,
        work,
    ) = profile_components(flow, admitted)
    return MulticommodityFlowProfileResult._from_kernel(
        flow,
        divergences=divergences,
        edge_profiles=edge_profiles,
        all_demands_routed=all_demands_routed,
        capacity_feasible=capacity_feasible,
        congestion=congestion,
        work=work,
    )


def check_multicommodity_flow_witness(
    flow: MulticommodityFlow,
) -> MulticommodityFlowWitnessCheckResult:
    """Exactly replay one submitted tensor and return its feasibility verdict.

    The admitted profile kernel computes the commodity-vertex divergences and
    aggregate edge loads once, in exact rational arithmetic.  This adapter then
    classifies those already-exact components against the declared terminals
    and capacities; it introduces no new arithmetic beyond exact comparisons
    and never consults a floating-point solver.
    """

    profile = compute_multicommodity_flow_profile(flow)
    expected: dict[tuple[str, int], Fraction] = {}
    for commodity in flow.commodities:
        demand = commodity.demand.as_fraction()
        for vertex in range(flow.network.vertex_count):
            if vertex == commodity.source:
                expected[(commodity.commodity_id, vertex)] = demand
            elif vertex == commodity.sink:
                expected[(commodity.commodity_id, vertex)] = -demand
            else:
                expected[(commodity.commodity_id, vertex)] = Fraction(0)

    violating_commodities: set[str] = set()
    violating_vertices: list[CommodityVertexViolation] = []
    for row in profile.divergences:
        target = expected[(row.commodity_id, row.vertex)]
        if row.divergence.as_fraction() != target:
            violating_commodities.add(row.commodity_id)
            violating_vertices.append(
                CommodityVertexViolation(
                    commodity_id=row.commodity_id,
                    vertex=row.vertex,
                    divergence=row.divergence,
                    expected=CanonicalRational.from_fraction(target),
                )
            )

    violating_edges: list[EdgeCapacityViolation] = []
    for edge, edge_profile in zip(
        flow.network.edges, profile.edge_profiles, strict=True
    ):
        if edge_profile.load.as_fraction() > edge.capacity.as_fraction():
            violating_edges.append(
                EdgeCapacityViolation(
                    source=edge.source,
                    target=edge.target,
                    load=edge_profile.load,
                    capacity=edge.capacity,
                )
            )

    return MulticommodityFlowWitnessCheckResult._from_kernel(
        flow,
        divergences=profile.divergences,
        edge_profiles=profile.edge_profiles,
        all_demands_routed=profile.all_demands_routed,
        capacity_feasible=profile.capacity_feasible,
        congestion=profile.congestion,
        violating_commodities=tuple(sorted(violating_commodities)),
        violating_vertices=tuple(violating_vertices),
        violating_edges=tuple(violating_edges),
        work=profile.work,
    )


__all__ = [
    "check_multicommodity_flow_witness",
    "compute_multicommodity_flow_profile",
]
