"""Public operation adapters for exact multicommodity-flow profiles."""

from __future__ import annotations

from jacobian.math.graphs.multicommodity_flow._kernel import profile_components
from jacobian.math.graphs.multicommodity_flow._models import (
    AdmittedProfileScan,
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
    its own measured scan, so a native call executes exactly the two
    charged passes. Parsed MCP requests reuse their parse-time scan instead.
    """

    return _profile_result(flow, None)


def _profile_result(
    flow: MulticommodityFlow,
    admitted: AdmittedProfileScan | None,
) -> MulticommodityFlowProfileResult:
    (
        divergences,
        edge_profiles,
        all_demands_routed,
        capacity_feasible,
        congestion,
        work,
    ) = profile_components(flow, admitted)
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
    """Run one parsed MCP request through the native profile computation.

    Request validation already performed and admitted the operation's single
    component scan; it is reused here as the producer pass, so execution
    adds only the independent replay pass.
    """

    return _profile_result(request.flow, request._admitted_scan)


__all__ = ["compute_multicommodity_flow_profile"]
