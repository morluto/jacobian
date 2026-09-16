"""Polytopal-complex operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.polytopes.complexes._models import (
    PolytopalComplexClosureRequest,
    PolytopalComplexClosureResult,
)
from jacobian.math.geometry.polytopes.complexes.operations import (
    polytopal_complex_closure,
)


def compute_polytopal_complex_closure(
    request: PolytopalComplexClosureRequest,
) -> PolytopalComplexClosureResult:
    """Unpack a request and project the native face-closure result."""
    return polytopal_complex_closure(request.cells)


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
)


__all__ = ["TOOLS"]
