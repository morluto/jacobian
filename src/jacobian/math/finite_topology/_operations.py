"""Wire adapters for public finite-topology operations."""

from __future__ import annotations

from jacobian.math.finite_topology._models import (
    BeatPointsRequest,
    BeatPointsResult,
    ConnectedComponentsRequest,
    ConnectedComponentsResult,
    ContinuityRequest,
    ContinuityResult,
    SpecializationPreorderRequest,
    SpecializationPreorderResult,
)
from jacobian.math.finite_topology.operations import (
    beat_points,
    connected_components,
    continuity,
    specialization_preorder,
)


def compute_specialization_preorder(
    request: SpecializationPreorderRequest,
) -> SpecializationPreorderResult:
    return SpecializationPreorderResult._from_kernel(
        request, specialization_preorder(request.topology)
    )


def compute_connected_components(
    request: ConnectedComponentsRequest,
) -> ConnectedComponentsResult:
    components = connected_components(request.topology)
    return ConnectedComponentsResult._from_kernel(request, components)


def compute_continuity(request: ContinuityRequest) -> ContinuityResult:
    analysis = continuity(request.domain, request.codomain, request.point_map)
    return ContinuityResult._from_kernel(
        request,
        is_continuous=analysis.is_continuous,
        violating_open_set=analysis.violating_open_set,
        violating_preimage=analysis.violating_preimage,
    )


def compute_beat_points(request: BeatPointsRequest) -> BeatPointsResult:
    analysis = beat_points(request.topology)
    return BeatPointsResult._from_kernel(
        request,
        down_beat_points=analysis.down_beat_points,
        up_beat_points=analysis.up_beat_points,
    )


def verify_specialization_preorder_result(
    result: SpecializationPreorderResult,
) -> bool:
    """Verify one claimed specialization relation in the admitted topology envelope."""

    return result.relation == specialization_preorder(result.topology)


def verify_connected_components_result(result: ConnectedComponentsResult) -> bool:
    """Verify one claimed component partition in the admitted topology envelope."""

    components = connected_components(result.topology)
    return result.components == components and result.component_count == len(components)


def verify_continuity_result(result: ContinuityResult) -> bool:
    """Verify one claimed continuity analysis in the admitted topology envelope."""

    analysis = continuity(result.domain, result.codomain, result.point_map)
    return (
        result.is_continuous == analysis.is_continuous
        and result.violating_open_set == analysis.violating_open_set
        and result.violating_preimage == analysis.violating_preimage
    )


def verify_beat_points_result(result: BeatPointsResult) -> bool:
    """Verify one claimed beat-point analysis in the admitted topology envelope."""

    analysis = beat_points(result.topology)
    return (
        result.down_beat_points == analysis.down_beat_points
        and result.up_beat_points == analysis.up_beat_points
    )


__all__ = [
    "compute_beat_points",
    "compute_connected_components",
    "compute_continuity",
    "compute_specialization_preorder",
    "verify_beat_points_result",
    "verify_connected_components_result",
    "verify_continuity_result",
    "verify_specialization_preorder_result",
]
