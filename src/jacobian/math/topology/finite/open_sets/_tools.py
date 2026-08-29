"""Public finite-topology operation declarations."""

from __future__ import annotations

from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationDomainValidationError
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
    try:
        analysis = continuity(request.domain, request.codomain, request.point_map)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("point_map",),
            code="finite_topology.map_carrier_mismatch",
            message=str(exc),
        ) from exc
    return ContinuityResult._from_kernel(
        request,
        is_continuous=analysis.is_continuous,
        violating_open_set=analysis.violating_open_set,
        violating_preimage=analysis.violating_preimage,
    )


def compute_beat_points(request: BeatPointsRequest) -> BeatPointsResult:
    _admit_topology(request.topology, location=("topology",))
    try:
        analysis = beat_points(request.topology)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("topology",),
            code="finite_topology.beat_points_require_t0",
            message=str(exc),
        ) from exc
    return BeatPointsResult._from_kernel(
        request,
        down_beat_points=analysis.down_beat_points,
        up_beat_points=analysis.up_beat_points,
    )


_SIERPINSKI = {
    "point_count": 2,
    "open_sets": [[], [1], [0, 1]],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="topology.specialization_preorder.compute",
        title="Compute a specialization preorder",
        description=(
            "Compute the complete specialization relation, explicitly oriented so "
            "relation[x,y] means x lies in the closure of the singleton y."
        ),
        request_type=SpecializationPreorderRequest,
        result_type=SpecializationPreorderResult,
        run=compute_specialization_preorder,
        tags=("topology", "finite-topology", "specialization", "exact"),
        examples=(
            example(
                "sierpinski_specialization",
                "Compute the specialization preorder of the Sierpinski space.",
                {"topology": _SIERPINSKI},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.connected_components.compute",
        title="Compute finite-space connected components",
        description=(
            "Compute the complete component partition of a finite space through "
            "the undirected comparability graph of its specialization preorder."
        ),
        request_type=ConnectedComponentsRequest,
        result_type=ConnectedComponentsResult,
        run=compute_connected_components,
        tags=("topology", "finite-topology", "connected-components", "exact"),
        examples=(
            example(
                "sierpinski_components",
                "Compute the components of the connected Sierpinski space.",
                {"topology": _SIERPINSKI},
            ),
        ),
    ),
    MathTool(
        operation_id="topology.is_continuous.compute",
        title="Decide continuity of a finite-space map",
        description=(
            "Check every codomain open-set preimage and return the first exact "
            "counterexample when the point map is not continuous."
        ),
        request_type=ContinuityRequest,
        result_type=ContinuityResult,
        run=compute_continuity,
        tags=("topology", "finite-topology", "continuity", "exact"),
        examples=(
            example(
                "sierpinski_identity",
                "Check the identity map of the Sierpinski space.",
                {
                    "domain": _SIERPINSKI,
                    "codomain": _SIERPINSKI,
                    "point_map": {
                        "domain_point_count": 2,
                        "codomain_point_count": 2,
                        "values": [0, 1],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="topology.beat_points.compute",
        title="Compute beat points of a finite T0 space",
        description=(
            "Compute every up and down beat point in the strict specialization "
            "order, together with its unique extremum witness."
        ),
        request_type=BeatPointsRequest,
        result_type=BeatPointsResult,
        run=compute_beat_points,
        tags=("topology", "finite-topology", "beat-points", "exact", "t0"),
        examples=(
            example(
                "sierpinski_beat_points",
                "Compute beat points and witnesses in the Sierpinski space.",
                {"topology": _SIERPINSKI},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
