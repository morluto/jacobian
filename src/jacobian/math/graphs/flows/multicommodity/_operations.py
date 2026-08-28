"""Public operation adapters for exact multicommodity-flow profiles."""

from __future__ import annotations

from jacobian.math.graphs.flows.multicommodity._kernel import profile_components
from jacobian.math.graphs.flows.multicommodity._models import (
    AdmittedProfileScan,
    MulticommodityFlow,
    MulticommodityFlowProfileRequest,
    MulticommodityFlowProfileResult,
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


def _run_multicommodity_flow_profile(
    request: MulticommodityFlowProfileRequest,
) -> MulticommodityFlowProfileResult:
    """Run one parsed MCP request through the native profile computation.

    The wire request is structural. Native execution performs semantic
    admission once and passes the resulting reusable scan to the kernel.
    """

    return _profile_result(
        request.flow, _require_profile_output_admission(request.flow)
    )


__all__ = ["compute_multicommodity_flow_profile"]
