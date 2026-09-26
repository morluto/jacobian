"""Public exact graph projections of polytopal complexes."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.polytopes.complexes.adjacency._models import (
    PolytopalAdjacencyRequest,
    PolytopalComplexAdjacencyGraph,
)
from jacobian.math.geometry.polytopes.complexes.adjacency.operations import (
    MAX_POLYTOPAL_ADJACENCY_RESULT_COORDINATES,
    MAX_POLYTOPAL_ADJACENCY_RESULT_DIGITS,
    polytopal_complex_adjacency_graph,
)

_TRIANGLE_PAIR = {
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
                    "vertex_id": "a",
                    "coordinates": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                    ],
                },
                {
                    "vertex_id": "b",
                    "coordinates": [
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
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
    ]
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polyhedral_complex.adjacency_graph.compute",
        title="Compute maximal-cell facet adjacency in a rational polytopal complex",
        description=(
            "Construct the complete exact face-to-face complex of the supplied "
            "rational maximal cells, then return a canonical simple undirected "
            "graph whose vertices are the canonical maximal-cell IDs, bound to "
            "their exact cell vertices. An edge "
            "joins exactly two cells whose intersection is a codimension-one "
            "face, labelled by the exact shared face and its coordinates; "
            "intersections only along lower-dimensional faces do not produce "
            "edges. The request uses the face-closure bounds of at most "
            "16 cells in ambient dimension at most 4, and the graph is bounded "
            "by 16 vertices, 120 edges, "
            f"{MAX_POLYTOPAL_ADJACENCY_RESULT_COORDINATES} coordinate values, "
            f"and {MAX_POLYTOPAL_ADJACENCY_RESULT_DIGITS} exact digits."
        ),
        request_type=PolytopalAdjacencyRequest,
        result_type=PolytopalComplexAdjacencyGraph,
        run=lambda request: polytopal_complex_adjacency_graph(request.cells),
        tags=("polytope", "polytopal-complex", "adjacency", "exact-rational"),
        discovery_terms=(
            "polytopal complex maximal-cell adjacency graph",
            "which polytopes share a facet",
            "facet adjacency graph of rational cells",
        ),
        examples=(
            OperationExample(
                name="triangles_sharing_a_full_edge",
                description=(
                    "Two triangles that meet along their complete diagonal edge "
                    "produce one adjacency edge."
                ),
                input=_TRIANGLE_PAIR,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
