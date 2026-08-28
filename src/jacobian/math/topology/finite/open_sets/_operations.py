"""Wire adapters for public finite-topology operations."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.finite.open_sets._models import (
    MAX_TOPOLOGY_OPERATION_OPENS,
    MAX_TOPOLOGY_OPERATION_POINTS,
    BeatPointsRequest,
    BeatPointsResult,
    ConnectedComponentsRequest,
    ConnectedComponentsResult,
    ContinuityRequest,
    ContinuityResult,
    SpecializationPreorderRequest,
    SpecializationPreorderResult,
)
from jacobian.math.topology.finite.open_sets.operations import (
    beat_points,
    connected_components,
    continuity,
    specialization_preorder,
)
from jacobian.math.topology.finite.open_sets.values import FiniteTopology


def _admit_topology(
    topology: FiniteTopology, *, location: tuple[str | int, ...]
) -> None:
    if topology.point_count > MAX_TOPOLOGY_OPERATION_POINTS:
        raise OperationDomainValidationError(
            location=location,
            code="finite_topology.point_count_exceeds_bound",
            message=(
                "finite-topology operations support at most "
                f"{MAX_TOPOLOGY_OPERATION_POINTS} points"
            ),
        )
    if len(topology.open_sets) > MAX_TOPOLOGY_OPERATION_OPENS:
        raise OperationDomainValidationError(
            location=location,
            code="finite_topology.open_set_count_exceeds_bound",
            message=(
                "finite-topology operations support at most "
                f"{MAX_TOPOLOGY_OPERATION_OPENS} open sets"
            ),
        )


def compute_specialization_preorder(
    request: SpecializationPreorderRequest,
) -> SpecializationPreorderResult:
    _admit_topology(request.topology, location=("topology",))
    return SpecializationPreorderResult._from_kernel(
        request, specialization_preorder(request.topology)
    )


def compute_connected_components(
    request: ConnectedComponentsRequest,
) -> ConnectedComponentsResult:
    _admit_topology(request.topology, location=("topology",))
    components = connected_components(request.topology)
    return ConnectedComponentsResult._from_kernel(request, components)


def compute_continuity(request: ContinuityRequest) -> ContinuityResult:
    _admit_topology(request.domain, location=("domain",))
    _admit_topology(request.codomain, location=("codomain",))
    analysis = continuity(request.domain, request.codomain, request.point_map)
    return ContinuityResult._from_kernel(
        request,
        is_continuous=analysis.is_continuous,
        violating_open_set=analysis.violating_open_set,
        violating_preimage=analysis.violating_preimage,
    )


def compute_beat_points(request: BeatPointsRequest) -> BeatPointsResult:
    _admit_topology(request.topology, location=("topology",))
    analysis = beat_points(request.topology)
    return BeatPointsResult._from_kernel(
        request,
        down_beat_points=analysis.down_beat_points,
        up_beat_points=analysis.up_beat_points,
    )


__all__ = [
    "compute_beat_points",
    "compute_connected_components",
    "compute_continuity",
    "compute_specialization_preorder",
]
