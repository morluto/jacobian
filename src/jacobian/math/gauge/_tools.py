"""Public declaration for exact lattice-gauge path holonomy."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.gauge._models import (
    GaugeTransformRequest,
    GaugeTransformResult,
    HolonomyRequest,
    HolonomyResult,
    PlaquetteRequest,
    PlaquetteResult,
)
from jacobian.math.gauge.operations import (
    gauge_transform,
    path_holonomy,
    plaquette_curvature,
)


def _run_transform(request: GaugeTransformRequest) -> GaugeTransformResult:
    return gauge_transform(request.field, request.vertex_values)


def _run_plaquette(request: PlaquetteRequest) -> PlaquetteResult:
    return plaquette_curvature(request.field, request.path)


def _run_holonomy(request: HolonomyRequest) -> HolonomyResult:
    return path_holonomy(request.field, request.path)


_TRIANGLE_FIELD = {
    "lattice": {
        "vertices": ["a", "b", "c"],
        "edges": [
            {"edge_id": "ab", "tail": "a", "head": "b"},
            {"edge_id": "bc", "tail": "b", "head": "c"},
            {"edge_id": "ca", "tail": "c", "head": "a"},
        ],
    },
    "degree": 3,
    "edge_labels": [
        {"edge_id": "ab", "label": {"degree": 3, "image": [1, 2, 0]}},
        {"edge_id": "bc", "label": {"degree": 3, "image": [1, 2, 0]}},
        {"edge_id": "ca", "label": {"degree": 3, "image": [1, 2, 0]}},
    ],
}
_TRIANGLE_PATH = {
    "steps": [
        {"edge_id": "ab", "forward": True},
        {"edge_id": "bc", "forward": True},
        {"edge_id": "ca", "forward": True},
    ]
}

TOOLS = (
    MathTool(
        operation_id="lattice_gauge.transform.compute",
        title="Apply an exact finite gauge transformation to an edge field",
        description=(
            "Apply the typed nonabelian transformation U'_e=h_tail^-1 U_e h_head "
            "to every edge of a permutation-valued lattice field, retaining the "
            "source lattice, group degree, and complete vertex-frame map."
        ),
        request_type=GaugeTransformRequest,
        result_type=GaugeTransformResult,
        run=_run_transform,
        tags=("lattice-gauge", "gauge-transform", "nonabelian", "exact"),
        examples=(
            OperationExample(
                name="triangle_vertex_gauge_transform",
                description="Transform a triangle edge field by exact vertex frames; every lattice vertex must have one frame element.",
                input={
                    "field": _TRIANGLE_FIELD,
                    "vertex_values": [
                        {"vertex": "a", "value": {"degree": 3, "image": [1, 2, 0]}},
                        {"vertex": "b", "value": {"degree": 3, "image": [0, 1, 2]}},
                        {"vertex": "c", "value": {"degree": 3, "image": [0, 1, 2]}},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lattice_gauge.plaquette.compute",
        title="Compute exact oriented plaquette curvature",
        description=(
            "Compute the ordered nonabelian holonomy around a closed oriented "
            "lattice plaquette path; reversing orientation gives the exact inverse "
            "and the result remains bound to the field and path."
        ),
        request_type=PlaquetteRequest,
        result_type=PlaquetteResult,
        run=_run_plaquette,
        tags=("lattice-gauge", "plaquette", "curvature", "exact"),
        examples=(
            OperationExample(
                name="triangle_plaquette",
                description="Compute triangle plaquette curvature; the path must be closed and chain over the field lattice.",
                input={"field": _TRIANGLE_FIELD, "path": _TRIANGLE_PATH},
            ),
        ),
    ),
    MathTool(
        operation_id="lattice_gauge.holonomy.compute",
        title="Compute the ordered exact holonomy of a lattice gauge field",
        description=(
            "For a finite oriented gauge lattice with permutation-valued edge "
            "labels and an oriented edge path chaining head-to-tail, return "
            "the ordered exact group product Hol = U_{e_1} ... U_{e_m} under "
            "the left-to-right convention with the per-edge contribution map "
            "and start/end vertices. Backward steps resolve to exact inverses, "
            "so reverse traversal gives the inverse holonomy."
        ),
        request_type=HolonomyRequest,
        result_type=HolonomyResult,
        run=_run_holonomy,
        tags=("lattice-gauge", "holonomy", "exact"),
        discovery_terms=(
            "lattice gauge path holonomy",
            "Wilson line ordered product",
            "gauge field edge product",
        ),
        examples=(
            OperationExample(
                name="triangle_plaquette_holonomy",
                description=(
                    "Compute the exact S3 holonomy around an oriented triangle; "
                    "path steps must chain head-to-tail over labelled edges."
                ),
                input={
                    "field": {
                        "lattice": {
                            "vertices": ["a", "b", "c"],
                            "edges": [
                                {"edge_id": "ab", "tail": "a", "head": "b"},
                                {"edge_id": "bc", "tail": "b", "head": "c"},
                                {"edge_id": "ca", "tail": "c", "head": "a"},
                            ],
                        },
                        "degree": 3,
                        "edge_labels": [
                            {
                                "edge_id": "ab",
                                "label": {"degree": 3, "image": [1, 2, 0]},
                            },
                            {
                                "edge_id": "bc",
                                "label": {"degree": 3, "image": [1, 2, 0]},
                            },
                            {
                                "edge_id": "ca",
                                "label": {"degree": 3, "image": [1, 2, 0]},
                            },
                        ],
                    },
                    "path": {
                        "steps": [
                            {"edge_id": "ab", "forward": True},
                            {"edge_id": "bc", "forward": True},
                            {"edge_id": "ca", "forward": True},
                        ]
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
