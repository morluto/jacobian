"""Public operation adapters for exact multicommodity-flow profiles."""

from __future__ import annotations

from jacobian.math.graphs.multicommodity_flow._kernel import profile_components
from jacobian.math.graphs.multicommodity_flow._models import (
    MulticommodityFlow,
    MulticommodityFlowProfileRequest,
    MulticommodityFlowProfileResult,
    _require_profile_output_admission,
)


def compute_multicommodity_flow_profile(
    flow: MulticommodityFlow,
) -> MulticommodityFlowProfileResult:
    """Compute the exact bounded load and conservation profile of one tensor."""

    # The canonical tensor carries only representation bounds, so this native
    # boundary enforces the profile execution envelope itself. The MCP path
    # has already admitted its flow once during request parsing.
    _require_profile_output_admission(flow)
    return _admitted_profile_result(flow)


def _admitted_profile_result(
    flow: MulticommodityFlow,
) -> MulticommodityFlowProfileResult:
    (
        divergences,
        edge_profiles,
        all_demands_routed,
        capacity_feasible,
        congestion,
        work,
    ) = profile_components(flow)
    return MulticommodityFlowProfileResult(
        flow=flow,
        divergences=divergences,
        edge_profiles=edge_profiles,
        all_demands_routed=all_demands_routed,
        capacity_feasible=capacity_feasible,
        congestion=congestion,
        work=work,
    )


def _run_multicommodity_flow_profile(
    request: MulticommodityFlowProfileRequest,
) -> MulticommodityFlowProfileResult:
    """Run one parsed MCP request through the native profile computation."""

    return _admitted_profile_result(request.flow)


__all__ = ["compute_multicommodity_flow_profile"]
