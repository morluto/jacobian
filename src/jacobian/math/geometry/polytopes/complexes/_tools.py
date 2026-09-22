"""Polytopal-complex operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.polytopes.complexes._models import (
    PiecewiseEvaluationRequest,
    PiecewiseEvaluationResult,
    PiecewisePolynomialRequest,
    PiecewisePolynomialResult,
    PolytopalComplexClosureRequest,
    PolytopalComplexClosureResult,
    SplineSpaceRequest,
    SplineSpaceResult,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    piecewise_polynomial_evaluate,
    piecewise_polynomial_from_maximal_pieces,
    polytopal_complex_closure,
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


def _run_eval(request: Any) -> Any:
    return piecewise_polynomial_evaluate(request.function, request.point)


def _run_spline(request: Any) -> Any:
    return spline_space(request.complex, request.degree, request.smoothness)


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
)

__all__ = ["TOOLS"]
