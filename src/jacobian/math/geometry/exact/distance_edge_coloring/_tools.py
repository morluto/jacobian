"""Exact distance graph operation declaration."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.geometry.exact.distance_edge_coloring._models import (
    DistanceEdgeColoringRequest,
    DistanceEdgeColoringResult,
)
from jacobian.math.geometry.exact.distance_edge_coloring.operations import (
    compute_distance_edge_coloring,
)


def _compute(request: DistanceEdgeColoringRequest) -> DistanceEdgeColoringResult:
    return compute_distance_edge_coloring(request.configuration)


TOOLS: MathTools = (
    MathTool(
        operation_id="geometry.points.distance_edge_coloring.compute",
        title="Construct the exact distance-coloured complete graph",
        description=(
            "Return the complete graph on labelled rational points, represented "
            "as a source-bound 2-uniform FiniteHypergraph with explicit edge IDs. "
            "Its sorted rational palette contains every distinct squared Euclidean "
            "distance; edge colour indices agree exactly when lengths agree. "
            "Coincident points retain zero-distance edges. The indexed colouring "
            "composes with same-colour union conflicts and rainbow subsets. "
            "Admits 2-64 points in 1-20 dimensions, at most 2016 edges, 8 million "
            "source coefficient bits, 64 million source/palette bits and 10^13 "
            "bit-arithmetic work units; exact growth must fit rational carriers."
        ),
        request_type=DistanceEdgeColoringRequest,
        result_type=DistanceEdgeColoringResult,
        run=_compute,
        tags=("geometry", "exact", "graph", "coloring"),
        examples=(
            OperationExample(
                name="unit_square",
                description="Four sides have squared distance 1; two diagonals have 2.",
                input={
                    "configuration": {
                        "points": [
                            {
                                "label": label,
                                "coordinates": [
                                    {"num": str(x), "den": "1"},
                                    {"num": str(y), "den": "1"},
                                ],
                            }
                            for label, x, y in (
                                ("a", 0, 0),
                                ("b", 1, 0),
                                ("c", 0, 1),
                                ("d", 1, 1),
                            )
                        ]
                    }
                },
            ),
        ),
    ),
)
