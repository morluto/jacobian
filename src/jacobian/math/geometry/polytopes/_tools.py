"""Polytope operation ownership and declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.polytopes._models import (
    FacetIncidenceRequest,
    FacetIncidenceResult,
    PolytopeSupportRequest,
    PolytopeSupportResult,
    PolytopeVolumeRequest,
    PolytopeVolumeResult,
    RationalVPolytope,
)
from jacobian.math.geometry.polytopes.operations import (
    facet_incidence,
    polytope_support,
    polytope_volume,
)
from jacobian.math.geometry.polytopes.values import Vertex


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


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "polytope.rational.support.compute",
        "Compute an exact rational polytope support value",
        "For a full-dimensional rational polytope with one labelled coordinate "
        "axis and a complete irredundant V-representation, compute the exact "
        "support value h_P(u)=max_{x in P}<u,x> and return every maximizing "
        "vertex as the complete exposed face. The exact support kernel is one "
        "bounded vertex-by-covector pass; the V-value separately proves that "
        "each supplied generator is an extreme vertex before evaluation.",
        PolytopeSupportRequest,
        PolytopeSupportResult,
        compute_polytope_support,
        "polytope",
        "support-function",
        "exposed-face",
        "exact-rational",
        examples=(
            example(
                "unit_square_top_edge",
                "Unit square on axes [x, y]; the covector (0,1) exposes the "
                "complete top edge. The covector's serialized space must be "
                "identical to the polytope's: same axis labels, same order.",
                {
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
    _op(
        "polytope.facets.compute",
        "Compute the complete exact facet-incidence profile of a rational polytope",
        "Compute every maximal supporting facet of the convex hull of an ordered "
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
        FacetIncidenceRequest,
        FacetIncidenceResult,
        compute_facet_incidence,
        "polytope",
        "facets",
        "incidence",
        "exact-rational",
        examples=(
            example(
                "unit_square",
                "Compute the four supporting facets of the unit square and their "
                "source-row incidences; the four supplied points affinely span R^2.",
                {
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
    _op(
        "polytope.volume.compute",
        "Compute the exact rational volume of a bounded polytope",
        "Compute the exact rational volume of a bounded rational polytope "
        "from its V-representation — bare vertices or an unchanged "
        "labelled ``RationalVPolytope`` value such as a support result's "
        "``polytope`` — or H-representation (half-spaces) for ambient "
        "dimension d <= 6, via triangulation and SymPy exact "
        "determinant-based simplex volume. Every half-space must carry a "
        "nonzero normal: rows whose coefficients are all zero are rejected.",
        PolytopeVolumeRequest,
        PolytopeVolumeResult,
        compute_polytope_volume,
        "polytope",
        "volume",
        "exact-rational",
        examples=(
            example(
                "unit_cube_vertices",
                "Unit cube [0,1]^2 split into two triangles (volume = 1).",
                {
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
            example(
                "unit_square_halfspaces",
                "Unit square [0,1]^2 as four half-spaces, each with a "
                "nonzero normal (volume = 1).",
                {
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
)


__all__ = ["TOOLS"]
