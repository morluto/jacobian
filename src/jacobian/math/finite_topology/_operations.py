"""Domain adapter for finite topology operations."""

from __future__ import annotations

from jacobian.math.finite_topology._models import (
    BeatPointsRequest,
    BeatPointsResult,
    ClosureRequest,
    ClosureResult,
    ConnectedComponentsRequest,
    ConnectedComponentsResult,
    InteriorRequest,
    InteriorResult,
    IsContinuousRequest,
    IsContinuousResult,
    SpecializationPreorderRequest,
    SpecializationPreorderResult,
)
from jacobian.math.finite_topology.operations import (
    beat_points,
    closure,
    connected_components,
    interior,
    is_continuous,
    specialization_preorder,
)

__all__ = [
    "compute_beat_points",
    "compute_closure",
    "compute_connected_components",
    "compute_interior",
    "compute_is_continuous",
    "compute_specialization_preorder",
]


def compute_specialization_preorder(
    request: SpecializationPreorderRequest,
) -> SpecializationPreorderResult:
    return SpecializationPreorderResult(
        preorder=specialization_preorder(request.topology)
    )


def compute_closure(request: ClosureRequest) -> ClosureResult:
    return ClosureResult(closure=tuple(sorted(closure(request.topology, request.subset))))


def compute_interior(request: InteriorRequest) -> InteriorResult:
    return InteriorResult(interior=tuple(sorted(interior(request.topology, request.subset))))


def compute_connected_components(
    request: ConnectedComponentsRequest,
) -> ConnectedComponentsResult:
    return ConnectedComponentsResult(
        components=connected_components(request.topology)
    )


def compute_is_continuous(request: IsContinuousRequest) -> IsContinuousResult:
    return IsContinuousResult(
        is_continuous=is_continuous(
            request.domain, request.codomain, request.function
        )
    )


def compute_beat_points(request: BeatPointsRequest) -> BeatPointsResult:
    down, up = beat_points(request.topology)
    return BeatPointsResult(up_beat_points=up, down_beat_points=down)
