"""Public operation adapters for exact multicommodity-flow profiles."""

from __future__ import annotations

from jacobian.math.graphs.multicommodity_flow._kernel import profile_components
from jacobian.math.graphs.multicommodity_flow._models import (
    MulticommodityFlowProfileRequest,
    MulticommodityFlowProfileResult,
)


def compute_multicommodity_flow_profile(
    request: MulticommodityFlowProfileRequest,
) -> MulticommodityFlowProfileResult:
    """Compute the exact bounded load and conservation profile of one tensor."""

    (
        divergences,
        edge_profiles,
        all_demands_routed,
        capacity_feasible,
        congestion,
        work,
    ) = profile_components(request.flow)
    return MulticommodityFlowProfileResult(
        flow=request.flow,
        divergences=divergences,
        edge_profiles=edge_profiles,
        all_demands_routed=all_demands_routed,
        capacity_feasible=capacity_feasible,
        congestion=congestion,
        work=work,
    )


__all__ = ["compute_multicommodity_flow_profile"]
