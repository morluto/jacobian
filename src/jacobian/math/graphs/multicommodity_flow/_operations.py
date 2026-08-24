"""Public operation adapters for exact multicommodity-flow profiles."""

from __future__ import annotations

from jacobian.math.graphs.multicommodity_flow._kernel import profile_components
from jacobian.math.graphs.multicommodity_flow._models import (
    MulticommodityFlow,
    MulticommodityFlowProfileRequest,
    MulticommodityFlowProfileResult,
)


def compute_multicommodity_flow_profile(
    flow: MulticommodityFlow,
) -> MulticommodityFlowProfileResult:
    """Compute the exact bounded load and conservation profile of one tensor.

    The canonical tensor value carries only representation bounds; this
    execution boundary admits the profile work and result envelope inside
    the kernel's single measured scan, so a native call executes exactly the
    two charged passes. Parsed MCP requests were already validated with the
    same admission at request parsing.
    """

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

    return compute_multicommodity_flow_profile(request.flow)


__all__ = ["compute_multicommodity_flow_profile"]
