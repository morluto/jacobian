"""Polytopal-complex operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.polytopes.complexes._models import (
    CommonRefinementRequest,
    CommonRefinementResult,
    GlobalPolynomialProfileRequest,
    GlobalPolynomialProfileResult,
    PiecewiseEvaluationRequest,
    PiecewiseEvaluationResult,
    PiecewisePolynomialAdditionRequest,
    PiecewisePolynomialMultiplicationRequest,
    PiecewisePolynomialRequest,
    PiecewisePolynomialResult,
    PiecewiseSmoothnessRequest,
    PiecewiseSmoothnessResult,
    PolytopalComplexAffineTransformRequest,
    PolytopalComplexAffineTransformResult,
    PolytopalComplexClosureRequest,
    PolytopalComplexClosureResult,
    SplineDimensionRequest,
    SplineDimensionResult,
    SplineEvaluationRequest,
    SplineEvaluationResult,
    SplineSpaceRequest,
    SplineSpaceResult,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    piecewise_polynomial_add,
    piecewise_polynomial_evaluate,
    piecewise_polynomial_from_maximal_pieces,
    piecewise_polynomial_global_profile,
    piecewise_polynomial_multiply,
    piecewise_polynomial_smoothness,
    polytopal_complex_affine_transform,
    polytopal_complex_closure,
    polytopal_complex_common_refinement,
    spline_dimension,
    spline_evaluate,
    spline_space,
)


def compute_polytopal_complex_closure(
    request: PolytopalComplexClosureRequest,
) -> PolytopalComplexClosureResult:
    """Unpack a request and project the native face-closure result."""
    return polytopal_complex_closure(request.cells)


_COMPLEX: dict[str, Any] = {
    "space": {"axes": ["x"]},
    "dimension": 1,
    "faces": [
        {"face_id": "f0", "dimension": -1, "vertices": [], "maximal_cell_ids": ["M0"]},
        {
            "face_id": "f1",
            "dimension": 0,
            "vertices": [{"coordinates": [{"num": "0", "den": "1"}]}],
            "maximal_cell_ids": ["M0"],
        },
        {
            "face_id": "f2",
            "dimension": 0,
            "vertices": [{"coordinates": [{"num": "1", "den": "1"}]}],
            "maximal_cell_ids": ["M0"],
        },
        {
            "face_id": "f3",
            "dimension": 1,
            "vertices": [
                {"coordinates": [{"num": "0", "den": "1"}]},
                {"coordinates": [{"num": "1", "den": "1"}]},
            ],
            "maximal_cell_ids": ["M0"],
        },
    ],
    "maximal_cells": [
        {
            "cell_id": "M0",
            "source_indices": [0],
            "dimension": 1,
            "vertices": [
                {"coordinates": [{"num": "0", "den": "1"}]},
                {"coordinates": [{"num": "1", "den": "1"}]},
            ],
            "facet_face_ids": ["f1", "f2"],
        }
    ],
    "cover_relations": [
        {"lower_face_id": "f0", "upper_face_id": "f1"},
        {"lower_face_id": "f0", "upper_face_id": "f2"},
        {"lower_face_id": "f1", "upper_face_id": "f3"},
        {"lower_face_id": "f2", "upper_face_id": "f3"},
    ],
    "pairwise_intersections": [],
    "source_cell_map": [{"source_index": 0, "cell_id": "M0"}],
    "f_vector": [1, 2, 1],
    "euler_characteristic": 1,
    "reduced_euler_characteristic": 0,
    "component_count": 1,
    "empty_face_admitted": True,
}
_POLY = {
    "domain": "QQ",
    "variables": ["x"],
    "polynomial": {
        "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}]
    },
}


def _run_piecewise(request: Any) -> Any:
    return piecewise_polynomial_from_maximal_pieces(request.complex, request.pieces)


def _run_piecewise_add(
    request: PiecewisePolynomialAdditionRequest,
) -> PiecewisePolynomialResult:
    return piecewise_polynomial_add(request)


def _run_piecewise_multiply(
    request: PiecewisePolynomialMultiplicationRequest,
) -> PiecewisePolynomialResult:
    return piecewise_polynomial_multiply(request)


def _run_eval(request: Any) -> Any:
    return piecewise_polynomial_evaluate(request.function, request.point)


def _run_smoothness(request: PiecewiseSmoothnessRequest) -> PiecewiseSmoothnessResult:
    return piecewise_polynomial_smoothness(request)


def _run_spline(request: Any) -> Any:
    return spline_space(request.complex, request.degree, request.smoothness)


def _run_spline_eval(request: SplineEvaluationRequest) -> SplineEvaluationResult:
    return spline_evaluate(request)


def _run_spline_dimension(
    request: SplineDimensionRequest,
) -> SplineDimensionResult:
    return spline_dimension(request)


def _run_global_polynomial_profile(
    request: GlobalPolynomialProfileRequest,
) -> GlobalPolynomialProfileResult:
    return piecewise_polynomial_global_profile(request)


def _run_common_refinement(request: CommonRefinementRequest) -> CommonRefinementResult:
    return polytopal_complex_common_refinement(request.left, request.right)


def _run_affine_transform(
    request: PolytopalComplexAffineTransformRequest,
) -> PolytopalComplexAffineTransformResult:
    return polytopal_complex_affine_transform(
        request.complex, request.matrix, request.translation
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polytopal_complex.closure.compute",
        title="Compute the exact face-closed rational polytopal complex of maximal cells",
        description=(
            "Given a finite nonempty family of bounded rational polytope "
            "presentations in one shared labelled ambient coordinate space, "
            "compute the canonical face-closed polytopal complex. Each cell's "
            "complete face family comes from intersecting it with its exact "
            "primitive supporting facet hyperplanes, and every pair of maximal "
            "cells is intersected exactly: each intersection must be empty or a "
            "face of both, otherwise the request is rejected as a typed "
            "non-face-to-face obstruction naming the offending presentation "
            "pair. Duplicate geometries are merged with presentation provenance "
            "retained; source-to-canonical transport, codimension-one cover "
            "relations, pairwise-intersection accounting, the f-vector, and both "
            "Euler characteristics are returned. The empty face is admitted with "
            "dimension -1 and the f-vector is indexed (f_{-1}, f_0, ..., f_d). "
            "At most 16 maximal cells in ambient dimension at most 4, with at "
            "most 1024 canonical faces, are admitted; pairwise-intersection and "
            "face-closure work is bounded before materialization. Only exact "
            "rational arithmetic is used."
        ),
        request_type=PolytopalComplexClosureRequest,
        result_type=PolytopalComplexClosureResult,
        run=compute_polytopal_complex_closure,
        tags=("polytope", "polytopal-complex", "incidence", "exact-rational"),
        discovery_terms=(
            "polytopal complex",
            "face-closed polyhedral complex",
            "common refinement",
            "cell complex",
            "face closure",
            "f-vector",
        ),
        examples=(
            OperationExample(
                name="two_triangles_share_edge",
                description=(
                    "Two rational triangles in the shared labelled plane [x, y] "
                    "meeting along the diagonal edge from (1, 0) to (0, 1); the "
                    "closure identifies that edge once, giving f-vector "
                    "(1, 4, 5, 2) and Euler characteristic 1. Both cells must "
                    "share one coordinate space."
                ),
                input={
                    "cells": [
                        {
                            "space": {"axes": ["x", "y"]},
                            "vertices": [
                                {
                                    "vertex_id": "a",
                                    "coordinates": [
                                        {"num": "0", "den": "1"},
                                        {"num": "0", "den": "1"},
                                    ],
                                },
                                {
                                    "vertex_id": "b",
                                    "coordinates": [
                                        {"num": "1", "den": "1"},
                                        {"num": "0", "den": "1"},
                                    ],
                                },
                                {
                                    "vertex_id": "c",
                                    "coordinates": [
                                        {"num": "0", "den": "1"},
                                        {"num": "1", "den": "1"},
                                    ],
                                },
                            ],
                        },
                        {
                            "space": {"axes": ["x", "y"]},
                            "vertices": [
                                {
                                    "vertex_id": "d",
                                    "coordinates": [
                                        {"num": "1", "den": "1"},
                                        {"num": "0", "den": "1"},
                                    ],
                                },
                                {
                                    "vertex_id": "e",
                                    "coordinates": [
                                        {"num": "1", "den": "1"},
                                        {"num": "1", "den": "1"},
                                    ],
                                },
                                {
                                    "vertex_id": "f",
                                    "coordinates": [
                                        {"num": "0", "den": "1"},
                                        {"num": "1", "den": "1"},
                                    ],
                                },
                            ],
                        },
                    ]
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polytopal_complex.affine_transform.compute",
        title="Apply an exact invertible affine map to a polytopal complex",
        description=(
            "Apply x -> A*x+b to every face and maximal cell of a canonical "
            "rational polytopal complex. A must be square and nonsingular over "
            "QQ. The result returns the canonical transformed complex and "
            "complete source-bound bijections on faces and maximal cells; "
            "dimensions, f-vector, cover relations, pairwise intersections, and "
            "cell-facet incidence are preserved. Work, rational-coordinate "
            "growth, and complete serialized output are admitted before geometry "
            "is rebuilt. The labelled coordinate axes are unchanged."
        ),
        request_type=PolytopalComplexAffineTransformRequest,
        result_type=PolytopalComplexAffineTransformResult,
        run=_run_affine_transform,
        tags=("geometry", "polytopal-complex", "affine-map", "exact-rational"),
        discovery_terms=(
            "affine transform polytopal complex",
            "transport polyhedral complex",
            "affine image of a polytopal complex",
        ),
        examples=(
            OperationExample(
                name="affine_transform_segment",
                description=(
                    "Scale the unit segment by two and translate it by three; "
                    "the resulting segment has endpoints 3 and 5."
                ),
                input={
                    "complex": _COMPLEX,
                    "matrix": [[{"num": "2", "den": "1"}]],
                    "translation": [{"num": "3", "den": "1"}],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polytopal_complex.common_refinement.compute",
        title="Compute the exact common refinement of two rational complexes",
        description=(
            "Intersect every pair of maximal cells from two canonical bounded "
            "full-dimensional rational polytopal complexes in the same labelled "
            "affine space. "
            "The operation requires equal supports, proved by exact volume "
            "conservation on every source cell, and returns the canonical "
            "face-closed overlay plus source-cell-pair provenance. Ambient "
            "dimension is at most three, each side has at most eight cells, "
            "each cell has at most sixteen vertices, and at most sixteen "
            "overlay cells are admitted."
        ),
        request_type=CommonRefinementRequest,
        result_type=CommonRefinementResult,
        run=_run_common_refinement,
        tags=("polytope", "polytopal-complex", "refinement", "exact-rational"),
        discovery_terms=(
            "common refinement of polytopal complexes",
            "overlay cell decomposition",
        ),
        examples=(
            OperationExample(
                name="same_segment_refinement",
                description="Overlay a segment complex with itself; its sole cell maps to the same overlay cell.",
                input={"left": _COMPLEX, "right": _COMPLEX},
            ),
        ),
    ),
    MathTool(
        operation_id="piecewise_polynomial.add.compute",
        title="Add compatible piecewise-polynomial functions exactly",
        description=(
            "Add two compatible scalar QQ-valued piecewise-polynomial functions "
            "on the identical canonical rational complex. The inputs and result "
            "are C0 functions; polynomial total degree of the sum is at most "
            "the maximum input degree. Input continuity is recomputed from the "
            "cell pieces. The output retains exact per-cell polynomials and the "
            "complete shared-face compatibility profile. Each result piece is "
            "bounded by 4096 terms and aggregate rational output by 64 MiB."
        ),
        request_type=PiecewisePolynomialAdditionRequest,
        result_type=PiecewisePolynomialResult,
        run=_run_piecewise_add,
        tags=("geometry", "piecewise-polynomial", "addition", "exact-rational"),
        discovery_terms=(
            "add piecewise-polynomial functions",
            "sum compatible polynomial pieces",
        ),
        examples=(
            OperationExample(
                name="add_constants_on_segment",
                description="Sum two constant compatible functions on the same exact segment complex.",
                input={
                    "left": {
                        "complex": _COMPLEX,
                        "pieces": [{"cell_id": "M0", "polynomial": _POLY}],
                        "compatibility": [],
                        "status": "COMPATIBLE",
                    },
                    "right": {
                        "complex": _COMPLEX,
                        "pieces": [
                            {
                                "cell_id": "M0",
                                "polynomial": {
                                    "domain": "QQ",
                                    "variables": ["x"],
                                    "polynomial": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "2", "den": "1"},
                                                "exponents": [0],
                                            }
                                        ]
                                    },
                                },
                            }
                        ],
                        "compatibility": [],
                        "status": "COMPATIBLE",
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="piecewise_polynomial.multiply.compute",
        title="Multiply compatible piecewise-polynomial functions exactly",
        description=(
            "Multiply two scalar QQ-valued C0 piecewise-polynomial functions "
            "on the identical canonical rational complex and ordered coordinate "
            "ring. The result has per-cell products of degree at most the sum "
            "of the input degrees, with its shared-face zero compatibility "
            "ledger derived from recomputed input ledgers. Admission bounds "
            "aggregate convolution work, coefficient growth, support and the "
            "complete canonical result at 10 MiB before polynomial expansion."
        ),
        request_type=PiecewisePolynomialMultiplicationRequest,
        result_type=PiecewisePolynomialResult,
        run=_run_piecewise_multiply,
        tags=("geometry", "piecewise-polynomial", "multiplication", "exact-rational"),
        discovery_terms=(
            "multiply piecewise-polynomial functions",
            "product of compatible polynomial pieces",
        ),
        examples=(
            OperationExample(
                name="multiply_linear_functions_on_segment",
                description="Multiply x by 1+x on the same exact segment complex.",
                input={
                    "left": {
                        "complex": _COMPLEX,
                        "pieces": [
                            {
                                "cell_id": "M0",
                                "polynomial": {
                                    "domain": "QQ",
                                    "variables": ["x"],
                                    "polynomial": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "1", "den": "1"},
                                                "exponents": [1],
                                            }
                                        ]
                                    },
                                },
                            }
                        ],
                        "compatibility": [],
                        "status": "COMPATIBLE",
                    },
                    "right": {
                        "complex": _COMPLEX,
                        "pieces": [
                            {
                                "cell_id": "M0",
                                "polynomial": {
                                    "domain": "QQ",
                                    "variables": ["x"],
                                    "polynomial": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "1", "den": "1"},
                                                "exponents": [1],
                                            },
                                            {
                                                "coefficient": {"num": "1", "den": "1"},
                                                "exponents": [0],
                                            },
                                        ]
                                    },
                                },
                            }
                        ],
                        "compatibility": [],
                        "status": "COMPATIBLE",
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="piecewise_polynomial.from_maximal_pieces.compute",
        title="Construct an exact compatible piecewise-polynomial function",
        description="Reduce every maximal-cell polynomial difference modulo each shared-face affine ideal and return the complete compatibility profile; pieces must use the complex coordinate ring.",
        request_type=PiecewisePolynomialRequest,
        result_type=PiecewisePolynomialResult,
        run=_run_piecewise,
        tags=("geometry", "piecewise-polynomial"),
        discovery_terms=("piecewise polynomial continuity", "face ideal compatibility"),
        examples=(
            OperationExample(
                name="constant_segment_piece",
                description="Construct the constant piece on one segment; the polynomial ring must use the segment axis.",
                input={
                    "complex": _COMPLEX,
                    "pieces": [{"cell_id": "M0", "polynomial": _POLY}],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="piecewise_polynomial.evaluate.compute",
        title="Evaluate a compatible piecewise-polynomial function exactly",
        description="Evaluate all containing maximal-cell pieces at a rational point and return one common exact value or OUTSIDE_SUPPORT.",
        request_type=PiecewiseEvaluationRequest,
        result_type=PiecewiseEvaluationResult,
        run=_run_eval,
        tags=("geometry", "piecewise-polynomial", "evaluation"),
        discovery_terms=("piecewise polynomial evaluation", "common face evaluation"),
        examples=(
            OperationExample(
                name="segment_midpoint",
                description="Evaluate the constant piece at the midpoint of the segment; the point must use the complex axis.",
                input={
                    "function": {
                        "complex": _COMPLEX,
                        "pieces": [{"cell_id": "M0", "polynomial": _POLY}],
                        "compatibility": [],
                        "status": "COMPATIBLE",
                    },
                    "point": {"coordinates": [{"num": "1", "den": "2"}]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="piecewise_polynomial.global_polynomial_profile.compute",
        title="Check whether compatible pieces are one ambient polynomial",
        description=(
            "For a continuous piecewise-polynomial function on a complex whose "
            "maximal cells are all full-dimensional, determine whether every "
            "cell carries the same ambient rational polynomial. Full dimension "
            "makes restriction injective, so exact coefficient comparison is "
            "complete. The source function and global polynomial are retained; "
            "lower-dimensional maximal cells are rejected because they do not "
            "determine an ambient extension."
        ),
        request_type=GlobalPolynomialProfileRequest,
        result_type=GlobalPolynomialProfileResult,
        run=_run_global_polynomial_profile,
        tags=("geometry", "piecewise-polynomial", "global-polynomial"),
        discovery_terms=(
            "is a piecewise polynomial globally polynomial",
            "single ambient polynomial profile",
        ),
        examples=(
            OperationExample(
                name="constant_segment_is_global",
                description="The same constant on every full-dimensional cell is one global polynomial.",
                input={
                    "function": {
                        "complex": _COMPLEX,
                        "pieces": [{"cell_id": "M0", "polynomial": _POLY}],
                        "compatibility": [],
                        "status": "COMPATIBLE",
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polyhedral_complex.spline_dimension.compute",
        title="Compute the exact dimension of a bounded rational spline space",
        description=(
            "Build the complete exact shared-facet compatibility matrix for "
            "the requested degree and smoothness, compute its rank and nullity, "
            "and return the source-bound matrix and coefficient axis without "
            "materializing a nullspace basis. Matrix size, exact rank work, "
            "scalar growth, and serialized output are admitted independently "
            "from the full spline-space operation."
        ),
        request_type=SplineDimensionRequest,
        result_type=SplineDimensionResult,
        run=_run_spline_dimension,
        tags=("geometry", "spline", "dimension", "exact-rational"),
        discovery_terms=("spline dimension", "dimension of a spline space"),
        examples=(
            OperationExample(
                name="segment_constants_dimension",
                description=(
                    "Compute the degree-zero C0 spline dimension on one segment; "
                    "the one unconstrained coefficient gives dimension one."
                ),
                input={"complex": _COMPLEX, "degree": 0, "smoothness": 0},
            ),
        ),
    ),
    MathTool(
        operation_id="polyhedral_complex.spline_space.compute",
        title="Compute a bounded exact rational spline nullspace",
        description="Build the complete shared-facet compatibility matrix for degree and C^r scalar splines on a pure rational complex, then return its exact rank, nullity, and nullspace basis.",
        request_type=SplineSpaceRequest,
        result_type=SplineSpaceResult,
        run=_run_spline,
        tags=("geometry", "spline", "nullspace"),
        discovery_terms=("polynomial spline space", "piecewise polynomial nullspace"),
        examples=(
            OperationExample(
                name="segment_constants",
                description="Compute the degree-zero C0 spline space on one segment; the complex is pure and the smoothness convention is exact.",
                input={"complex": _COMPLEX, "degree": 0, "smoothness": 0},
            ),
        ),
    ),
    MathTool(
        operation_id="polyhedral_complex.spline.evaluate.compute",
        title="Evaluate an exact rational spline from basis coordinates",
        description=(
            "Compute the admitted C^r spline space for a face-closed rational "
            "polytopal complex, interpret basis_coefficients in the canonical "
            "nullspace basis returned by that space, and evaluate the resulting "
            "spline at one rational point. The point may lie on shared faces; "
            "C^0 continuity makes the exact values agree there. A point outside "
            "the union of maximal cells returns an empty containing-cell list and "
            "no value. Exact coefficient growth is admitted before evaluation. "
            "The same degree, smoothness, and complex envelope as the spline-space "
            "operation applies."
        ),
        request_type=SplineEvaluationRequest,
        result_type=SplineEvaluationResult,
        run=_run_spline_eval,
        tags=("geometry", "spline", "evaluation", "exact-rational"),
        discovery_terms=(
            "evaluate a polynomial spline",
            "spline nullspace basis coordinates",
            "exact spline point evaluation",
        ),
        examples=(
            OperationExample(
                name="constant_segment_spline_value",
                description=(
                    "Evaluate three times the sole degree-zero spline on the "
                    "unit segment at its midpoint."
                ),
                input={
                    "complex": _COMPLEX,
                    "degree": 0,
                    "smoothness": 0,
                    "basis_coefficients": [{"num": "3", "den": "1"}],
                    "point": {"coordinates": [{"num": "1", "den": "2"}]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="piecewise_polynomial.smoothness_profile.compute",
        title="Compute exact smoothness across piecewise-polynomial facets",
        description=(
            "For every interior facet between maximal cells, determine the "
            "largest requested C^r order by exact divisibility of the cell "
            "polynomial difference by the facet equation to power r+1. "
            "Returns -1 when the pieces are discontinuous and the minimum "
            "facet order as the global profile."
        ),
        request_type=PiecewiseSmoothnessRequest,
        result_type=PiecewiseSmoothnessResult,
        run=_run_smoothness,
        tags=("geometry", "piecewise-polynomial", "smoothness", "exact-rational"),
        discovery_terms=("piecewise polynomial smoothness", "C^r continuity profile"),
        examples=(
            OperationExample(
                name="single_cell_has_no_interior_facet_obstructions",
                description=(
                    "A polynomial on one cell has no interior facets, so its "
                    "requested smoothness profile reaches the supplied order."
                ),
                input={
                    "function": {
                        "complex": _COMPLEX,
                        "pieces": [
                            {
                                "cell_id": "M0",
                                "polynomial": {
                                    "domain": "QQ",
                                    "variables": ["x"],
                                    "polynomial": {
                                        "terms": [
                                            {
                                                "coefficient": {"num": "1", "den": "1"},
                                                "exponents": [1],
                                            }
                                        ]
                                    },
                                },
                            }
                        ],
                        "compatibility": [],
                        "status": "COMPATIBLE",
                    },
                    "max_smoothness": 4,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
