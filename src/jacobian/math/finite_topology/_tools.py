"""Public finite-topology operation declarations."""

from typing import Any

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
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
from jacobian.math.finite_topology._operations import (
    compute_beat_points,
    compute_connected_components,
    compute_continuity,
    compute_specialization_preorder,
)

_SIERPINSKI = {
    "point_count": 2,
    "open_sets": [[], [1], [0, 1]],
}

FINITE_TOPOLOGY_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="topology.specialization_preorder.compute",
        version="1",
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
        version="1",
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
        version="1",
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
        version="1",
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

TOOLS = FINITE_TOPOLOGY_OPERATIONS

__all__ = ["TOOLS"]
