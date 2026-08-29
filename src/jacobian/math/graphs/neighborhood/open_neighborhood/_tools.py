"""Typed declarations for the open neighbourhood operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.neighborhood.open_neighborhood._models import (
    OpenNeighborhoodRequest,
    OpenNeighborhoodResult,
)
from jacobian.math.graphs.neighborhood.open_neighborhood.operations import (
    compute_open_neighborhood,
)


def _compute(request: OpenNeighborhoodRequest) -> OpenNeighborhoodResult:
    return compute_open_neighborhood(request.graph, request.selected_vertices)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.neighborhood.open.compute",
        title="Compute the open neighbourhood of a selected vertex set",
        description=(
            "For one bounded simple undirected graph and one selected vertex "
            "set, return the exact sorted open neighbourhood: all vertices "
            "outside the set that are adjacent to at least one member of it."
        ),
        request_type=OpenNeighborhoodRequest,
        result_type=OpenNeighborhoodResult,
        run=_compute,
        tags=("graph", "neighborhood", "exact"),
        examples=(
            example(
                "path_graph",
                "Open neighbourhood of vertex b in a path P4.",
                {
                    "graph": {
                        "vertices": ["a", "b", "c", "d"],
                        "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
                    },
                    "selected_vertices": ["b"],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
