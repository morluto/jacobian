"""Polytope operation ownership and declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.polytopes._models import (
    EdgeProfileRequest,
    EdgeProfileResult,
    FacetIncidenceRequest,
    FacetIncidenceResult,
    JoinRequest,
    JoinResult,
    PolytopeFaceLatticeRequest,
    PolytopeFaceLatticeResult,
    PolytopeSupportRequest,
    PolytopeSupportResult,
    PolytopeVolumeRequest,
    PolytopeVolumeResult,
    PrismRequest,
    PrismResult,
    PyramidRequest,
    PyramidResult,
    RationalVPolytope,
    VertexFigureRequest,
    VertexFigureResult,
)
from jacobian.math.geometry.polytopes.operations import (
    facet_incidence,
    polytope_edge_profile,
    polytope_face_lattice,
    polytope_join,
    polytope_prism,
    polytope_pyramid,
    polytope_support,
    polytope_vertex_figure,
    polytope_volume,
)
from jacobian.math.geometry.polytopes.polyhedron_conversion import (
    halfspaces_to_v_presentation,
)
from jacobian.math.geometry.polytopes.values import (
    RationalHPolyhedron,
    RationalPolyhedronVPresentation,
    Vertex,
)


def compute_polytope_support(request: PolytopeSupportRequest) -> PolytopeSupportResult:
    """Unpack a request and project the native support result."""
    return polytope_support(request.polytope, request.covector)


def compute_facet_incidence(request: FacetIncidenceRequest) -> FacetIncidenceResult:
    """Unpack a request and project the native facet profile."""
    vertices = request.vertices
    if isinstance(vertices, RationalVPolytope):
        vertices = tuple(
            Vertex(coordinates=vertex.coordinates) for vertex in vertices.vertices
        )
    return facet_incidence(vertices, request.dimension_bound)


def compute_polytope_pyramid(request: PyramidRequest) -> PyramidResult:
    """Unpack a request and project the native pyramid result."""
    return polytope_pyramid(request.polytope, request.height_axis)


def compute_polytope_prism(request: PrismRequest) -> PrismResult:
    """Unpack a request and project the native prism result."""
    return polytope_prism(request.polytope, request.height_axis)


def compute_polytope_join(request: JoinRequest) -> JoinResult:
    """Unpack a request and project the native join result."""
    return polytope_join(request.left, request.right, request.height_axis)


def compute_polytope_edge_profile(request: EdgeProfileRequest) -> EdgeProfileResult:
    """Unpack a request and project the native edge-profile result."""
    return polytope_edge_profile(request.polytope, request.dimension_bound)


def compute_polytope_vertex_figure(request: VertexFigureRequest) -> VertexFigureResult:
    """Unpack a request and project the native vertex-figure result."""
    return polytope_vertex_figure(request.polytope, request.vertex_id)


def compute_polytope_face_lattice(
    request: PolytopeFaceLatticeRequest,
) -> PolytopeFaceLatticeResult:
    """Unpack a request and compute its exact rank-three face lattice."""
    return polytope_face_lattice(request.polytope)


def compute_polytope_volume(request: PolytopeVolumeRequest) -> PolytopeVolumeResult:
    """Unpack a request and project the native volume result."""
    vertices = request.vertices
    if isinstance(vertices, RationalVPolytope):
        vertices = tuple(
            Vertex(coordinates=vertex.coordinates) for vertex in vertices.vertices
        )
    return polytope_volume(
        vertices,
        request.halfspaces,
        request.dimension_bound,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polytope.face_lattice.compute",
        title="Compute the complete face lattice of a three-dimensional polytope",
        description=(
            "From an ordered, labelled rational V-representation in dimension three, "
            "recompute the complete exact facet incidence, identify the extreme "
            "source rows, and return every face including the empty and whole faces "
            "with all Hasse cover relations. Face labels are sorted source-vertex "
            "indices; redundant input rows remain in the source but are excluded "
            "from the face vertices. Admission bounds coordinate height, exact facet "
            "enumeration, postprocessing work, face and cover counts, and result size."
        ),
        request_type=PolytopeFaceLatticeRequest,
        result_type=PolytopeFaceLatticeResult,
        run=compute_polytope_face_lattice,
        tags=("polytope", "face-lattice", "face-poset", "exact-rational"),
        discovery_terms=(
            "face lattice of a polytope",
            "polytope face poset",
            "Hasse diagram of a three-dimensional polytope",
        ),
        examples=(
            OperationExample(
                name="tetrahedron_face_lattice",
                description=(
                    "The standard tetrahedron has four vertices, six edges, "
                    "four triangular facets, and its empty and whole faces."
                ),
                input={
                    "polytope": {
                        "space": {"axes": ["x", "y", "z"]},
                        "vertices": [
                            {
                                "vertex_id": "ex",
                                "coordinates": [
                                    {"num": "1", "den": "1"},
                                    {"num": "0", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                            },
                            {
                                "vertex_id": "ey",
                                "coordinates": [
                                    {"num": "0", "den": "1"},
                                    {"num": "1", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                            },
                            {
                                "vertex_id": "ez",
                                "coordinates": [
                                    {"num": "0", "den": "1"},
                                    {"num": "0", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ],
                            },
                            {
                                "vertex_id": "origin",
                                "coordinates": [
                                    {"num": "0", "den": "1"},
                                    {"num": "0", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                            },
                        ],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polytope.rational.h_to_v.compute",
        title="Convert an exact rational H-polyhedron to finite points and directions",
        description=(
            "Convert inequalities a·x <= b to a serializable exact V-presentation "
            "containing finite points, oriented recession rays, and lineality directions. "
            "The value distinguishes empty from nonempty affine, bounded, and unbounded "
            "polyhedra. Generators are exact but are not promised minimal or canonical. "
            "Work, coefficient growth, and a conservative result-byte bound are admitted "
            "before double-description expansion."
        ),
        request_type=RationalHPolyhedron,
        result_type=RationalPolyhedronVPresentation,
        run=halfspaces_to_v_presentation,
        tags=("polyhedron", "H-to-V", "exact-rational", "recession-cone"),
        discovery_terms=(
            "H to V polyhedron conversion",
            "rational polyhedron vertices rays lineality",
        ),
        examples=(
            OperationExample(
                name="unit_square",
                description="Convert the four defining inequalities of the unit square on ordered axes [x,y].",
                input={
                    "space": {"axes": ["x", "y"]},
                    "inequalities": [
                        {
                            "normal": [
                                {"num": "-1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            "bound": {"num": "0", "den": "1"},
                        },
                        {
                            "normal": [
                                {"num": "0", "den": "1"},
                                {"num": "-1", "den": "1"},
                            ],
                            "bound": {"num": "0", "den": "1"},
                        },
                        {
                            "normal": [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            "bound": {"num": "1", "den": "1"},
                        },
                        {
                            "normal": [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                            "bound": {"num": "1", "den": "1"},
                        },
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polytope.rational.support.compute",
        title="Compute an exact rational polytope support value",
        description="For a full-dimensional rational polytope with one labelled coordinate "
        "axis and a complete irredundant V-representation, compute the exact "
        "support value h_P(u)=max_{x in P}<u,x> and return every maximizing "
        "vertex as the complete exposed face. The exact support kernel is one "
        "bounded vertex-by-covector pass; the V-value separately proves that "
        "each supplied generator is an extreme vertex before evaluation.",
        request_type=PolytopeSupportRequest,
        result_type=PolytopeSupportResult,
        run=compute_polytope_support,
        tags=("polytope", "support-function", "exposed-face", "exact-rational"),
        examples=(
            OperationExample(
                name="unit_square_top_edge",
                description="Unit square on axes [x, y]; the covector (0,1) exposes the "
                "complete top edge. The covector's serialized space must be "
                "identical to the polytope's: same axis labels, same order.",
                input={
                    "polytope": {
                        "space": {"axes": ["x", "y"]},
                        "vertices": [
                            {
                                "vertex_id": "bottom_left",
                                "coordinates": [
                                    {"num": "0", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                            },
                            {
                                "vertex_id": "bottom_right",
                                "coordinates": [
                                    {"num": "1", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                            },
                            {
                                "vertex_id": "top_left",
                                "coordinates": [
                                    {"num": "0", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ],
                            },
                            {
                                "vertex_id": "top_right",
                                "coordinates": [
                                    {"num": "1", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ],
                            },
                        ],
                    },
                    "covector": {
                        "space": {"axes": ["x", "y"]},
                        "components": [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polytope.facets.compute",
        title="Compute the complete exact facet-incidence profile of a rational polytope",
        description="Compute every maximal supporting facet of the convex hull of an ordered "
        "rational V-representation — bare vertices or an unchanged labelled "
        "``RationalVPolytope`` value such as a support result's ``polytope`` "
        "(d <= 7); lower-dimensional hulls are "
        "rejected. Each facet returns its canonical primitive supporting "
        "inequality as the shared half-space value plus the complete "
        "source-row incidence. For d <= 6 each row composes verbatim into "
        "polytope.volume.compute; that consumer caps dimension at 6 and "
        "rejects d = 7 rows. Request admission materializes the complete "
        "bounded enumeration, enforcing the published facet and incidence "
        "result limits before a request is accepted; the exact bounded "
        "SymPy kernel then computes that source-bound profile.",
        request_type=FacetIncidenceRequest,
        result_type=FacetIncidenceResult,
        run=compute_facet_incidence,
        tags=("polytope", "facets", "incidence", "exact-rational"),
        examples=(
            OperationExample(
                name="unit_square",
                description="Compute the four supporting facets of the unit square and their "
                "source-row incidences; the four supplied points affinely span R^2.",
                input={
                    "vertices": [
                        {
                            "coordinates": [
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                            ]
                        },
                        {
                            "coordinates": [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ]
                        },
                        {
                            "coordinates": [
                                {"num": "1", "den": "1"},
                                {"num": "1", "den": "1"},
                            ]
                        },
                        {
                            "coordinates": [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ]
                        },
                    ]
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polytope.volume.compute",
        title="Compute the exact rational volume of a bounded polytope",
        description="Compute the exact rational volume of a bounded rational polytope "
        "from its V-representation — bare vertices or an unchanged "
        "labelled ``RationalVPolytope`` value such as a support result's "
        "``polytope`` — or H-representation (half-spaces) for ambient "
        "dimension d <= 6, via triangulation and SymPy exact "
        "determinant-based simplex volume. Every half-space must carry a "
        "nonzero normal: rows whose coefficients are all zero are rejected.",
        request_type=PolytopeVolumeRequest,
        result_type=PolytopeVolumeResult,
        run=compute_polytope_volume,
        tags=("polytope", "volume", "exact-rational"),
        examples=(
            OperationExample(
                name="unit_cube_vertices",
                description="Unit cube [0,1]^2 split into two triangles (volume = 1).",
                input={
                    "vertices": [
                        {
                            "coordinates": [
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                            ]
                        },
                        {
                            "coordinates": [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ]
                        },
                        {
                            "coordinates": [
                                {"num": "1", "den": "1"},
                                {"num": "1", "den": "1"},
                            ]
                        },
                        {
                            "coordinates": [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ]
                        },
                    ],
                },
            ),
            OperationExample(
                name="unit_square_halfspaces",
                description="Unit square [0,1]^2 as four half-spaces, each with a "
                "nonzero normal (volume = 1).",
                input={
                    "halfspaces": [
                        {
                            "coefficients": [
                                {"num": "-1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            "offset": {"num": "0", "den": "1"},
                        },
                        {
                            "coefficients": [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            "offset": {"num": "1", "den": "1"},
                        },
                        {
                            "coefficients": [
                                {"num": "0", "den": "1"},
                                {"num": "-1", "den": "1"},
                            ],
                            "offset": {"num": "0", "den": "1"},
                        },
                        {
                            "coefficients": [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                            "offset": {"num": "1", "den": "1"},
                        },
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polytope.rational.pyramid.compute",
        title="Compute the exact pyramid over a bounded rational polytope",
        description="Embed each base vertex p as (p, 0) on a fresh height axis and "
        "take the convex hull with the apex (0, ..., 0, 1), returning the exact "
        "pyramid V-polytope with base vertices keeping their source IDs, the "
        "reserved apex ID 'apex', explicit source-to-base transport, and the "
        "replayed dimension identity dim(pyramid) = dim(P) + 1. The source must "
        "be a nonempty labelled rational V-polytope; the height axis must be a "
        "fresh label and no source vertex may use the reserved apex ID.",
        request_type=PyramidRequest,
        result_type=PyramidResult,
        run=compute_polytope_pyramid,
        tags=("polytope", "pyramid", "exact-rational"),
        discovery_terms=(
            "pyramid over a polytope",
            "cone over a polytope",
            "join with a point",
        ),
        examples=(
            OperationExample(
                name="segment_pyramid_triangle",
                description="Build the exact triangle that is the pyramid over the unit "
                "segment on axis [x]; the height axis 'h' must be fresh and no "
                "source vertex may use the reserved apex ID.",
                input={
                    "polytope": {
                        "space": {"axes": ["x"]},
                        "vertices": [
                            {
                                "vertex_id": "left",
                                "coordinates": [{"num": "0", "den": "1"}],
                            },
                            {
                                "vertex_id": "right",
                                "coordinates": [{"num": "1", "den": "1"}],
                            },
                        ],
                    },
                    "height_axis": "h",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polytope.rational.prism.compute",
        title="Compute the exact prism over a bounded rational polytope",
        description="Embed each source vertex p as a bottom vertex (p, 0) and a top "
        "vertex (p, 1) on a fresh height axis, returning the exact prism "
        "V-polytope P x [0, 1] with suffixed bottom/top transport IDs and the "
        "replayed dimension identity dim(prism) = dim(P) + 1. The height axis "
        "must be fresh and the suffixed IDs must be distinct.",
        request_type=PrismRequest,
        result_type=PrismResult,
        run=compute_polytope_prism,
        tags=("polytope", "prism", "exact-rational"),
        discovery_terms=(
            "prism over a polytope",
            "product with an interval",
            "extrusion of a polytope",
        ),
        examples=(
            OperationExample(
                name="segment_prism_square",
                description="Build the exact unit square that is the prism over the unit "
                "segment on axis [x]; the height axis 'h' must be fresh.",
                input={
                    "polytope": {
                        "space": {"axes": ["x"]},
                        "vertices": [
                            {
                                "vertex_id": "left",
                                "coordinates": [{"num": "0", "den": "1"}],
                            },
                            {
                                "vertex_id": "right",
                                "coordinates": [{"num": "1", "den": "1"}],
                            },
                        ],
                    },
                    "height_axis": "h",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polytope.rational.join.compute",
        title="Compute the exact join of two bounded rational polytopes",
        description="Embed the left factor as (p, 0, 0) and the right factor as "
        "(0, q, 1) on (*left.axes, *right.axes, height_axis), returning the "
        "exact join V-polytope with unchanged source vertex IDs, explicit "
        "left/right transport, and the replayed dimension identity "
        "dim(join) = dim(P) + dim(Q) + 1. The factors must live on disjoint "
        "axes with disjoint vertex IDs and the height axis must be fresh.",
        request_type=JoinRequest,
        result_type=JoinResult,
        run=compute_polytope_join,
        tags=("polytope", "join", "exact-rational"),
        discovery_terms=(
            "join of polytopes",
            "convex join",
            "pyramid over two polytopes",
        ),
        examples=(
            OperationExample(
                name="segment_segment_join_tetrahedron",
                description="Build the exact tetrahedron that is the join of two unit "
                "segments on disjoint axes [x] and [y]; the height axis 'h' "
                "must be fresh and the factors must carry disjoint vertex IDs.",
                input={
                    "left": {
                        "space": {"axes": ["x"]},
                        "vertices": [
                            {
                                "vertex_id": "left_a",
                                "coordinates": [{"num": "0", "den": "1"}],
                            },
                            {
                                "vertex_id": "left_b",
                                "coordinates": [{"num": "1", "den": "1"}],
                            },
                        ],
                    },
                    "right": {
                        "space": {"axes": ["y"]},
                        "vertices": [
                            {
                                "vertex_id": "right_a",
                                "coordinates": [{"num": "0", "den": "1"}],
                            },
                            {
                                "vertex_id": "right_b",
                                "coordinates": [{"num": "1", "den": "1"}],
                            },
                        ],
                    },
                    "height_axis": "h",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polytope.rational.edge_profile.compute",
        title="Compute the exact edge graph of a bounded rational polytope",
        description="Compute every vertex-adjacency edge of a full-dimensional "
        "labelled rational V-polytope from its bounded facet profile: a pair "
        "is an edge exactly when the minimal face containing both (the "
        "intersection of their common facets) is one-dimensional with no "
        "third extreme vertex inside. Returns the sorted edge pairs over the "
        "exact extreme vertices with the replayed affine dimension. "
        "Redundant source rows carry no edges.",
        request_type=EdgeProfileRequest,
        result_type=EdgeProfileResult,
        run=compute_polytope_edge_profile,
        tags=("polytope", "edge-graph", "adjacency", "exact-rational"),
        discovery_terms=(
            "edge graph of a polytope",
            "vertex adjacency of a polytope",
            "one-skeleton of a polytope",
        ),
        examples=(
            OperationExample(
                name="unit_square_edges",
                description="Compute the four vertex-adjacency edges (the 4-cycle) "
                "of the unit square on axes [x, y].",
                input={
                    "polytope": {
                        "space": {"axes": ["x", "y"]},
                        "vertices": [
                            {
                                "vertex_id": "bottom_left",
                                "coordinates": [
                                    {"num": "0", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                            },
                            {
                                "vertex_id": "bottom_right",
                                "coordinates": [
                                    {"num": "1", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                            },
                            {
                                "vertex_id": "top_left",
                                "coordinates": [
                                    {"num": "0", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ],
                            },
                            {
                                "vertex_id": "top_right",
                                "coordinates": [
                                    {"num": "1", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ],
                            },
                        ],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polytope.rational.vertex_figure.compute",
        title="Compute the exact vertex figure of a polytope vertex",
        description="Compute the vertex figure at one extreme vertex as the "
        "convex hull of the edge-midpoints (v + u) / 2 over its edge "
        "neighbors u, returned as a V-polytope on the source axes with "
        "neighbor-to-figure transport (`sec_<neighbor_id>`) and the replayed "
        "midpoint and dimension identities dim(figure) = dim(P) - 1.",
        request_type=VertexFigureRequest,
        result_type=VertexFigureResult,
        run=compute_polytope_vertex_figure,
        tags=("polytope", "vertex-figure", "exact-rational"),
        discovery_terms=(
            "vertex figure of a polytope",
            "link of a polytope vertex",
            "section of a polytope at a vertex",
        ),
        examples=(
            OperationExample(
                name="square_vertex_figure_segment",
                description="Compute the vertex figure at the bottom_left corner "
                "of the unit square: the segment joining the midpoints of "
                "the two incident edges.",
                input={
                    "polytope": {
                        "space": {"axes": ["x", "y"]},
                        "vertices": [
                            {
                                "vertex_id": "bottom_left",
                                "coordinates": [
                                    {"num": "0", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                            },
                            {
                                "vertex_id": "bottom_right",
                                "coordinates": [
                                    {"num": "1", "den": "1"},
                                    {"num": "0", "den": "1"},
                                ],
                            },
                            {
                                "vertex_id": "top_left",
                                "coordinates": [
                                    {"num": "0", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ],
                            },
                            {
                                "vertex_id": "top_right",
                                "coordinates": [
                                    {"num": "1", "den": "1"},
                                    {"num": "1", "den": "1"},
                                ],
                            },
                        ],
                    },
                    "vertex_id": "bottom_left",
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
