"""Public declaration for exact lattice-gauge path holonomy."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.gauge._models import (
    FiniteGroupGaugeBasepointTransportRequest,
    FiniteGroupGaugeBasepointTransportResult,
    FiniteGroupGaugeComplex,
    FiniteGroupGaugeComplexRequest,
    FiniteGroupGaugeCurvatureRequest,
    FiniteGroupGaugeCurvatureResult,
    FiniteGroupGaugeHolonomyRequest,
    FiniteGroupGaugeHolonomyResult,
    FiniteGroupGaugeTransformRequest,
    FiniteGroupGaugeTransformResult,
    GaugeLoopFamilyHolonomies,
    GaugeLoopFamilyRequest,
    GaugeTransformRequest,
    GaugeTransformResult,
    HolonomyRequest,
    HolonomyResult,
    PermutationWilsonTraceRequest,
    PermutationWilsonTraceResult,
    PlaquetteRequest,
    PlaquetteResult,
)
from jacobian.math.gauge._su2_models import (
    SU2GaugeTransformRequest,
    SU2GaugeTransformResult,
    SU2HolonomyRequest,
    SU2HolonomyResult,
    SU2WilsonTraceResult,
)
from jacobian.math.gauge.finite_group import (
    finite_group_gauge_basepoint_transport,
    finite_group_gauge_curvature,
    finite_group_gauge_holonomy,
    finite_group_gauge_transform,
)
from jacobian.math.gauge.finite_group_complex import (
    construct_finite_group_gauge_complex,
)
from jacobian.math.gauge.observables import permutation_wilson_trace
from jacobian.math.gauge.operations import (
    gauge_transform,
    loop_family_holonomies,
    path_holonomy,
    plaquette_curvature,
)
from jacobian.math.gauge.su2 import (
    su2_gauge_transform,
    su2_path_holonomy,
    su2_wilson_trace,
)


def _run_transform(request: GaugeTransformRequest) -> GaugeTransformResult:
    return gauge_transform(request.field, request.vertex_values)


def _run_plaquette(request: PlaquetteRequest) -> PlaquetteResult:
    return plaquette_curvature(request.field, request.path)


def _run_holonomy(request: HolonomyRequest) -> HolonomyResult:
    return path_holonomy(request.field, request.path)


def _run_loop_family(
    request: GaugeLoopFamilyRequest,
) -> GaugeLoopFamilyHolonomies:
    return loop_family_holonomies(request.field, request.loops)


def _run_permutation_wilson(
    request: PermutationWilsonTraceRequest,
) -> PermutationWilsonTraceResult:
    return permutation_wilson_trace(request)


def _run_su2_transform(request: SU2GaugeTransformRequest) -> SU2GaugeTransformResult:
    return su2_gauge_transform(request)


def _run_su2_holonomy(request: SU2HolonomyRequest) -> SU2HolonomyResult:
    return su2_path_holonomy(request)


def _run_su2_wilson(request: SU2HolonomyResult) -> SU2WilsonTraceResult:
    return su2_wilson_trace(request)


def _run_finite_group_holonomy(
    request: FiniteGroupGaugeHolonomyRequest,
) -> FiniteGroupGaugeHolonomyResult:
    return finite_group_gauge_holonomy(request)


def _run_finite_group_basepoint_transport(
    request: FiniteGroupGaugeBasepointTransportRequest,
) -> FiniteGroupGaugeBasepointTransportResult:
    return finite_group_gauge_basepoint_transport(request)


def _run_finite_group_complex(
    request: FiniteGroupGaugeComplexRequest,
) -> FiniteGroupGaugeComplex:
    return construct_finite_group_gauge_complex(request)


def _run_finite_group_curvature(
    request: FiniteGroupGaugeCurvatureRequest,
) -> FiniteGroupGaugeCurvatureResult:
    return finite_group_gauge_curvature(request)


def _run_finite_group_transform(
    request: FiniteGroupGaugeTransformRequest,
) -> FiniteGroupGaugeTransformResult:
    return finite_group_gauge_transform(request)


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
_LOOP_FAMILY_FIELD = {
    "lattice": _TRIANGLE_FIELD["lattice"],
    "degree": 3,
    "edge_labels": [
        {"edge_id": "ab", "label": {"degree": 3, "image": [1, 0, 2]}},
        {"edge_id": "bc", "label": {"degree": 3, "image": [0, 2, 1]}},
        {"edge_id": "ca", "label": {"degree": 3, "image": [0, 1, 2]}},
    ],
}
_TRIVIAL_FIELD = {
    "lattice": {
        "vertices": ["v"],
        "edges": [{"edge_id": "loop", "tail": "v", "head": "v"}],
    },
    "degree": 1,
    "edge_labels": [{"edge_id": "loop", "label": {"degree": 1, "image": [0]}}],
}
_CYCLIC_THREE_GROUP = {
    "multiplication": [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
    "identity": 0,
    "inverse": [0, 2, 1],
}
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
_SU2_PLAQUETTE = {
    "coordinates": [
        {"num": "-140", "den": "221"},
        {"num": "-120", "den": "221"},
        {"num": "609", "den": "1105"},
        {"num": "12", "den": "1105"},
    ]
}
_SU2_HOLONOMY_INPUT = {
    "field": _SU2_FIELD,
    "path": _SU2_PATH,
    "holonomy": _SU2_PLAQUETTE,
    "start": "v0",
    "end": "v0",
}

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
        operation_id="lattice_gauge.permutation.wilson_trace.compute",
        title="Compute an exact permutation-representation Wilson trace",
        description=(
            "For a closed path over a finite permutation-valued gauge field, "
            "compute its ordered holonomy and the exact character of the "
            "natural degree-d permutation representation, equal to the number "
            "of fixed points of that holonomy."
        ),
        request_type=PermutationWilsonTraceRequest,
        result_type=PermutationWilsonTraceResult,
        run=_run_permutation_wilson,
        tags=("lattice-gauge", "wilson-loop", "character", "permutation", "exact"),
        discovery_terms=(
            "finite group Wilson loop character",
            "permutation representation trace of holonomy",
        ),
        examples=(
            OperationExample(
                name="triangle_permutation_wilson_trace",
                description=(
                    "The three equal 3-cycle link labels compose to the "
                    "identity, so the natural representation trace is three."
                ),
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
            "so reverse traversal gives the inverse holonomy. A zero-length "
            "path is the group identity at its explicitly named lattice vertex."
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
        operation_id="lattice_gauge.loop_family.holonomies.compute",
        title="Compute holonomies for an explicit finite family of loops",
        description=(
            "For one permutation-valued lattice gauge field and an explicit "
            "bounded family of based closed paths, return each exact holonomy "
            "with its path and basepoint. The result retains the source field "
            "once; no loop search or family generation is performed."
        ),
        request_type=GaugeLoopFamilyRequest,
        result_type=GaugeLoopFamilyHolonomies,
        run=_run_loop_family,
        tags=("lattice-gauge", "loop-family", "holonomy", "exact"),
        discovery_terms=(
            "finite family of lattice Wilson loops",
            "multiple loop holonomies",
            "lattice gauge loop profile",
        ),
        examples=(
            OperationExample(
                name="triangle_loop_family",
                description=(
                    "Evaluate the triangle loop and its reverse; the output "
                    "keeps the shared field once and records each basepoint."
                ),
                input={
                    "field": _LOOP_FAMILY_FIELD,
                    "loops": [
                        _TRIANGLE_PATH,
                        {
                            "steps": [
                                {"edge_id": "ca", "forward": False},
                                {"edge_id": "bc", "forward": False},
                                {"edge_id": "ab", "forward": False},
                            ]
                        },
                    ],
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
    ),
    MathTool(
        operation_id="lattice_gauge.finite_group.basepoint_transport.compute",
        title="Transport finite-group loop holonomy to another basepoint",
        description=(
            "Given a based loop and a connector from its basepoint to another "
            "vertex, return the transported loop reverse(connector) * loop * "
            "connector and its exact holonomy. The output records the identity "
            "Hol(connector)^-1 * Hol(loop) * Hol(connector) in one finite "
            "multiplication-table parent."
        ),
        request_type=FiniteGroupGaugeBasepointTransportRequest,
        result_type=FiniteGroupGaugeBasepointTransportResult,
        run=_run_finite_group_basepoint_transport,
        tags=("lattice-gauge", "finite-group", "basepoint", "holonomy", "exact"),
        discovery_terms=(
            "finite group loop basepoint transport",
            "conjugate holonomy along a path",
        ),
        examples=(
            OperationExample(
                name="s3_triangle_basepoint_transport",
                description=(
                    "Move the triangle loop at a to basepoint b along edge ab; "
                    "the loop holonomy is conjugated by the connector holonomy."
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
                        "group": {
                            "multiplication": [
                                [0, 1, 2, 3, 4, 5],
                                [1, 0, 3, 2, 5, 4],
                                [2, 4, 0, 5, 1, 3],
                                [3, 5, 1, 4, 0, 2],
                                [4, 2, 5, 0, 3, 1],
                                [5, 3, 4, 1, 2, 0],
                            ],
                            "identity": 0,
                            "inverse": [0, 1, 2, 4, 3, 5],
                        },
                        "edge_values": [
                            {
                                "edge_id": "ab",
                                "value": {
                                    "group": {
                                        "multiplication": [
                                            [0, 1, 2, 3, 4, 5],
                                            [1, 0, 3, 2, 5, 4],
                                            [2, 4, 0, 5, 1, 3],
                                            [3, 5, 1, 4, 0, 2],
                                            [4, 2, 5, 0, 3, 1],
                                            [5, 3, 4, 1, 2, 0],
                                        ],
                                        "identity": 0,
                                        "inverse": [0, 1, 2, 4, 3, 5],
                                    },
                                    "index": 1,
                                },
                            },
                            {
                                "edge_id": "bc",
                                "value": {
                                    "group": {
                                        "multiplication": [
                                            [0, 1, 2, 3, 4, 5],
                                            [1, 0, 3, 2, 5, 4],
                                            [2, 4, 0, 5, 1, 3],
                                            [3, 5, 1, 4, 0, 2],
                                            [4, 2, 5, 0, 3, 1],
                                            [5, 3, 4, 1, 2, 0],
                                        ],
                                        "identity": 0,
                                        "inverse": [0, 1, 2, 4, 3, 5],
                                    },
                                    "index": 2,
                                },
                            },
                            {
                                "edge_id": "ca",
                                "value": {
                                    "group": {
                                        "multiplication": [
                                            [0, 1, 2, 3, 4, 5],
                                            [1, 0, 3, 2, 5, 4],
                                            [2, 4, 0, 5, 1, 3],
                                            [3, 5, 1, 4, 0, 2],
                                            [4, 2, 5, 0, 3, 1],
                                            [5, 3, 4, 1, 2, 0],
                                        ],
                                        "identity": 0,
                                        "inverse": [0, 1, 2, 4, 3, 5],
                                    },
                                    "index": 0,
                                },
                            },
                        ],
                    },
                    "loop": {
                        "steps": [
                            {"edge_id": "ab", "forward": True},
                            {"edge_id": "bc", "forward": True},
                            {"edge_id": "ca", "forward": True},
                        ],
                        "basepoint": "a",
                    },
                    "connector": {
                        "steps": [{"edge_id": "ab", "forward": True}],
                        "basepoint": "a",
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lattice_gauge.finite_group.complex.construct.compute",
        title="Construct a source-bound finite-group gauge 2-complex",
        description=(
            "Retain oriented 2-cells as ordered closed attaching walks over one "
            "finite lattice and finite multiplication-table group. Empty walks "
            "name their attachment vertex; reversing orientation reverses the "
            "walk and each edge direction. This constructs topology for later "
            "curvature operations and does not compute curvature."
        ),
        request_type=FiniteGroupGaugeComplexRequest,
        result_type=FiniteGroupGaugeComplex,
        run=_run_finite_group_complex,
        tags=("lattice-gauge", "finite-group", "2-complex", "exact"),
        discovery_terms=(
            "oriented plaquette boundary for finite group gauge field",
            "finite lattice gauge 2-complex attaching word",
        ),
        examples=(
            OperationExample(
                name="loop_lattice_degenerate_faces",
                description=(
                    "A backtracking attaching walk and a constant attaching map "
                    "are distinct source-bound oriented 2-cells."
                ),
                input={
                    "lattice": {
                        "vertices": ["v"],
                        "edges": [{"edge_id": "loop", "tail": "v", "head": "v"}],
                    },
                    "group": {"multiplication": [[0]], "identity": 0, "inverse": [0]},
                    "faces": [
                        {
                            "face_id": "backtrack",
                            "boundary": {
                                "steps": [
                                    {"edge_id": "loop", "forward": True},
                                    {"edge_id": "loop", "forward": False},
                                ]
                            },
                        },
                        {
                            "face_id": "constant",
                            "boundary": {"steps": [], "basepoint": "v"},
                        },
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lattice_gauge.finite_group.gauge_transform.compute",
        title="Transform a finite-table lattice gauge field",
        description=(
            "Apply U'_(u->v)=g_u U_(u->v) g_v^-1 to every edge using one "
            "exact finite multiplication-table parent and one complete vertex "
            "frame map. Products follow the same left-to-right order as path "
            "holonomy; the result retains source, target, and canonical frames."
        ),
        request_type=FiniteGroupGaugeTransformRequest,
        result_type=FiniteGroupGaugeTransformResult,
        run=_run_finite_group_transform,
        tags=("lattice-gauge", "finite-group", "gauge-transform", "exact"),
        discovery_terms=(
            "finite group lattice gauge transformation",
            "transform table group edge field by vertex frames",
        ),
        examples=(
            OperationExample(
                name="cyclic_three_edge_frame_action",
                description=(
                    "On one edge in C3, the endpoint frames change link label "
                    "1 to 2 by g_tail U g_head^-1."
                ),
                input={
                    "field": {
                        "lattice": {
                            "vertices": ["u", "v"],
                            "edges": [{"edge_id": "uv", "tail": "u", "head": "v"}],
                        },
                        "group": _CYCLIC_THREE_GROUP,
                        "edge_values": [
                            {
                                "edge_id": "uv",
                                "value": {
                                    "group": _CYCLIC_THREE_GROUP,
                                    "index": 1,
                                },
                            }
                        ],
                    },
                    "vertex_values": [
                        {
                            "vertex": "u",
                            "value": {"group": _CYCLIC_THREE_GROUP, "index": 1},
                        },
                        {
                            "vertex": "v",
                            "value": {"group": _CYCLIC_THREE_GROUP, "index": 0},
                        },
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="lattice_gauge.finite_group.curvature.compute",
        title="Compute finite-group face curvature and flatness",
        description=(
            "Multiply the edge values along every oriented face attaching walk "
            "in its exact finite group table. Reversed steps use inverses, and "
            "flatness means every represented face product is the identity. The "
            "complex and field must share exactly the same lattice and group table."
        ),
        request_type=FiniteGroupGaugeCurvatureRequest,
        result_type=FiniteGroupGaugeCurvatureResult,
        run=_run_finite_group_curvature,
        tags=("lattice-gauge", "finite-group", "curvature", "flatness", "exact"),
        discovery_terms=(
            "finite group plaquette curvature",
            "flat finite group lattice gauge field",
            "2-complex face holonomy table group",
        ),
        examples=(
            OperationExample(
                name="cyclic_two_nonflat_triangle",
                description=(
                    "One nonidentity edge value gives nonidentity face curvature "
                    "in the cyclic group of order two."
                ),
                input={
                    "complex": {
                        "lattice": {
                            "vertices": ["a", "b", "c"],
                            "edges": [
                                {"edge_id": "ab", "tail": "a", "head": "b"},
                                {"edge_id": "bc", "tail": "b", "head": "c"},
                                {"edge_id": "ca", "tail": "c", "head": "a"},
                            ],
                        },
                        "group": {
                            "multiplication": [[0, 1], [1, 0]],
                            "identity": 0,
                            "inverse": [0, 1],
                        },
                        "faces": [
                            {
                                "face_id": "triangle",
                                "boundary": {
                                    "steps": [
                                        {"edge_id": "ab", "forward": True},
                                        {"edge_id": "bc", "forward": True},
                                        {"edge_id": "ca", "forward": True},
                                    ]
                                },
                            }
                        ],
                    },
                    "field": {
                        "lattice": {
                            "vertices": ["a", "b", "c"],
                            "edges": [
                                {"edge_id": "ab", "tail": "a", "head": "b"},
                                {"edge_id": "bc", "tail": "b", "head": "c"},
                                {"edge_id": "ca", "tail": "c", "head": "a"},
                            ],
                        },
                        "group": {
                            "multiplication": [[0, 1], [1, 0]],
                            "identity": 0,
                            "inverse": [0, 1],
                        },
                        "edge_values": [
                            {
                                "edge_id": "ab",
                                "value": {
                                    "group": {
                                        "multiplication": [[0, 1], [1, 0]],
                                        "identity": 0,
                                        "inverse": [0, 1],
                                    },
                                    "index": 1,
                                },
                            },
                            {
                                "edge_id": "bc",
                                "value": {
                                    "group": {
                                        "multiplication": [[0, 1], [1, 0]],
                                        "identity": 0,
                                        "inverse": [0, 1],
                                    },
                                    "index": 0,
                                },
                            },
                            {
                                "edge_id": "ca",
                                "value": {
                                    "group": {
                                        "multiplication": [[0, 1], [1, 0]],
                                        "identity": 0,
                                        "inverse": [0, 1],
                                    },
                                    "index": 0,
                                },
                            },
                        ],
                    },
                },
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
        discovery_terms=("SU(2) quaternion path holonomy",),
        examples=(
            OperationExample(
                name="rational_su2_plaquette_holonomy",
                description="Compose four rational unit-quaternion links around a closed square.",
                input={"field": _SU2_FIELD, "path": _SU2_PATH},
            ),
        ),
    ),
    MathTool(
        operation_id="lattice_gauge.su2.wilson_trace.compute",
        title="Compute the rational SU(2) Wilson trace",
        description=(
            "Return 2 Re(U) for a source-bound closed rational SU(2) "
            "holonomy as an exact canonical rational."
        ),
        request_type=SU2HolonomyResult,
        result_type=SU2WilsonTraceResult,
        run=_run_su2_wilson,
        tags=("lattice-gauge", "su2", "wilson-trace", "exact"),
        discovery_terms=("SU(2) Wilson loop trace",),
        examples=(
            OperationExample(
                name="rational_su2_wilson_trace",
                description="Compute the exact fundamental trace of a closed rational SU(2) holonomy.",
                input=_SU2_HOLONOMY_INPUT,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
