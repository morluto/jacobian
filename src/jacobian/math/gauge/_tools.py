"""Public declaration for exact lattice-gauge path holonomy."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.gauge._models import (
    FiniteGroupGaugeHolonomyRequest,
    FiniteGroupGaugeHolonomyResult,
    GaugeTransformRequest,
    GaugeTransformResult,
    HolonomyRequest,
    HolonomyResult,
    PlaquetteRequest,
    PlaquetteResult,
)
from jacobian.math.gauge._su2_models import (
    SU2GaugeTransformRequest,
    SU2GaugeTransformResult,
    SU2HolonomyRequest,
    SU2HolonomyResult,
)
from jacobian.math.gauge.finite_group import finite_group_gauge_holonomy
from jacobian.math.gauge.operations import (
    gauge_transform,
    path_holonomy,
    plaquette_curvature,
)
from jacobian.math.gauge.su2 import (
    su2_gauge_transform,
    su2_path_holonomy,
)


def _run_transform(request: GaugeTransformRequest) -> GaugeTransformResult:
    return gauge_transform(request.field, request.vertex_values)


def _run_plaquette(request: PlaquetteRequest) -> PlaquetteResult:
    return plaquette_curvature(request.field, request.path)


def _run_holonomy(request: HolonomyRequest) -> HolonomyResult:
    return path_holonomy(request.field, request.path)


def _run_finite_group_holonomy(
    request: FiniteGroupGaugeHolonomyRequest,
) -> FiniteGroupGaugeHolonomyResult:
    return finite_group_gauge_holonomy(request)


def _run_su2_transform(request: SU2GaugeTransformRequest) -> SU2GaugeTransformResult:
    return su2_gauge_transform(request.field, request.vertex_values)


def _run_su2_holonomy(request: SU2HolonomyRequest) -> SU2HolonomyResult:
    return su2_path_holonomy(request.field, request.path)


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
_TRIVIAL_FIELD = {
    "lattice": {
        "vertices": ["v"],
        "edges": [{"edge_id": "loop", "tail": "v", "head": "v"}],
    },
    "degree": 1,
    "edge_labels": [{"edge_id": "loop", "label": {"degree": 1, "image": [0]}}],
}
_CYCLIC_TWO_TABLE = {
    "multiplication": [[0, 1], [1, 0]],
    "identity": 0,
    "inverse": [0, 1],
}
_FINITE_GROUP_FIELD = {
    "lattice": {
        "vertices": ["a", "b"],
        "edges": [{"edge_id": "e1", "tail": "a", "head": "b"}],
    },
    "group": _CYCLIC_TWO_TABLE,
    "edge_values": [
        {"edge_id": "e1", "value": {"group": _CYCLIC_TWO_TABLE, "index": 1}},
    ],
}
_FINITE_GROUP_PATH = {"steps": [{"edge_id": "e1", "forward": True}]}
_SU2_IDENTITY = {
    "coordinates": [
        {"num": "1", "den": "1"},
        {"num": "0", "den": "1"},
        {"num": "0", "den": "1"},
        {"num": "0", "den": "1"},
    ]
}
_SU2_FIELD = {
    "lattice": {
        "vertices": ["v0", "v1", "v2", "v3"],
        "edges": [
            {"edge_id": "e0", "tail": "v0", "head": "v1"},
            {"edge_id": "e1", "tail": "v1", "head": "v2"},
            {"edge_id": "e2", "tail": "v2", "head": "v3"},
            {"edge_id": "e3", "tail": "v3", "head": "v0"},
        ],
    },
    "edge_values": [
        {
            "edge_id": "e0",
            "value": {
                "coordinates": [
                    {"num": "3", "den": "5"},
                    {"num": "4", "den": "5"},
                    {"num": "0", "den": "1"},
                    {"num": "0", "den": "1"},
                ]
            },
        },
        {
            "edge_id": "e1",
            "value": {
                "coordinates": [
                    {"num": "5", "den": "13"},
                    {"num": "0", "den": "1"},
                    {"num": "12", "den": "13"},
                    {"num": "0", "den": "1"},
                ]
            },
        },
        {
            "edge_id": "e2",
            "value": {
                "coordinates": [
                    {"num": "8", "den": "17"},
                    {"num": "0", "den": "1"},
                    {"num": "0", "den": "1"},
                    {"num": "15", "den": "17"},
                ]
            },
        },
        {
            "edge_id": "e3",
            "value": {
                "coordinates": [
                    {"num": "0", "den": "1"},
                    {"num": "1", "den": "1"},
                    {"num": "0", "den": "1"},
                    {"num": "0", "den": "1"},
                ]
            },
        },
    ],
}
_SU2_PATH = {"steps": [{"edge_id": f"e{i}", "forward": True} for i in range(4)]}

TOOLS = (
    MathTool(
        operation_id="lattice_gauge.transform.compute",
        title="Apply an exact finite gauge transformation to an edge field",
        description=(
            "Apply the typed nonabelian transformation U'_e=g_tail U_e g_head^-1 "
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
        operation_id="lattice_gauge.holonomy.compute",
        title="Compute the ordered exact holonomy of a lattice gauge field",
        description=(
            "For a finite oriented gauge lattice with permutation-valued edge "
            "labels and an oriented edge path chaining head-to-tail, return "
            "the ordered exact group product Hol = U_{e_1} ... U_{e_m} under "
            "the left-to-right convention with the per-edge contribution map "
            "and start/end vertices. Backward steps resolve to exact inverses, "
            "so reverse traversal gives the inverse holonomy. A zero-length "
            "path is the group identity at its explicitly named lattice vertex."
        ),
        request_type=HolonomyRequest,
        result_type=HolonomyResult,
        run=_run_holonomy,
        tags=("lattice-gauge", "holonomy", "plaquette", "curvature", "exact"),
        discovery_terms=(
            "lattice gauge path holonomy",
            "oriented plaquette curvature",
            "closed lattice gauge loop",
            "Wilson line ordered product",
            "gauge field edge product",
            "finite group Wilson loop character",
            "permutation representation trace of holonomy",
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
            OperationExample(
                name="trivial_group_identity_path",
                description=(
                    "A based zero-length path has identity holonomy and no edge "
                    "contributions; degree one is the trivial permutation group."
                ),
                input={
                    "field": _TRIVIAL_FIELD,
                    "path": {"steps": [], "basepoint": "v"},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lattice_gauge.finite_group.holonomy.compute",
        title="Compute path holonomy in an exact finite group table",
        description=(
            "Compose table-indexed edge values along an oriented lattice path. "
            "The field is bound to one complete finite multiplication table; "
            "backward steps use that table's inverse and factors multiply in "
            "path traversal order."
        ),
        request_type=FiniteGroupGaugeHolonomyRequest,
        result_type=FiniteGroupGaugeHolonomyResult,
        run=_run_finite_group_holonomy,
        tags=("lattice-gauge", "finite-group", "holonomy", "exact"),
        discovery_terms=(
            "finite group lattice gauge holonomy",
            "table group edge transport",
        ),
        examples=(
            OperationExample(
                name="cyclic_two_edge_holonomy",
                description=(
                    "Transport one non-identity C2 element along a single "
                    "oriented edge; the ordered product is that element."
                ),
                input={"field": _FINITE_GROUP_FIELD, "path": _FINITE_GROUP_PATH},
            ),
        ),
    ),
    MathTool(
        operation_id="lattice_gauge.su2.gauge_transform.compute",
        title="Transform a rational SU(2) gauge field",
        description=(
            "Apply U'_(u->v)=g_u U_(u->v) g_v^-1 to an exact rational "
            "unit-quaternion edge field; the source lattice and vertex frames "
            "remain bound to the result."
        ),
        request_type=SU2GaugeTransformRequest,
        result_type=SU2GaugeTransformResult,
        run=_run_su2_transform,
        tags=("lattice-gauge", "su2", "gauge-transform", "exact"),
        discovery_terms=("SU(2) quaternion lattice gauge transform",),
        examples=(
            OperationExample(
                name="rational_su2_plaquette_transform",
                description="Transform four exact rational SU(2) links by identity vertex frames.",
                input={
                    "field": _SU2_FIELD,
                    "vertex_values": [
                        {"vertex": f"v{i}", "value": _SU2_IDENTITY} for i in range(4)
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lattice_gauge.su2.holonomy.compute",
        title="Compute rational SU(2) path holonomy",
        description=(
            "Compose source-bound rational unit-quaternion edge values along "
            "an oriented lattice path; reverse traversal uses group inverse."
        ),
        request_type=SU2HolonomyRequest,
        result_type=SU2HolonomyResult,
        run=_run_su2_holonomy,
        tags=("lattice-gauge", "su2", "holonomy", "exact"),
        discovery_terms=(
            "SU(2) quaternion path holonomy",
            "SU(2) Wilson loop trace",
            "fundamental representation trace 2 Re of holonomy",
        ),
        examples=(
            OperationExample(
                name="rational_su2_plaquette_holonomy",
                description="Compose four rational unit-quaternion links around a closed square.",
                input={"field": _SU2_FIELD, "path": _SU2_PATH},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
