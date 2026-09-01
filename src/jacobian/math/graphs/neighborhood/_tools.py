"""Open-neighbourhood operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.neighborhood._bounds import (
    require_open_neighborhood_output_budget,
)
from jacobian.math.graphs.neighborhood._models import (
    NeighborhoodRequest,
    NeighborhoodResult,
)
from jacobian.math.graphs.neighborhood.operations import (
    open_neighborhood,
)


def compute_open_neighborhood(request: NeighborhoodRequest) -> NeighborhoodResult:
    result = open_neighborhood(request.graph, request.selected_vertices)
    require_open_neighborhood_output_budget(
        result.graph, result.selected_vertices, result.neighborhood
    )
    return result


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.neighborhood.open.compute",
        title="Compute the open neighbourhood of a selected vertex set",
        description=(
            "For a finite simple undirected graph G and a selected vertex set S, "
            "return the exact sorted open neighbourhood N_G(S) consisting of all "
            "vertices outside S adjacent to at least one member of S, in canonical "
            "source-vertex order."
        ),
        request_type=NeighborhoodRequest,
        result_type=NeighborhoodResult,
        run=compute_open_neighborhood,
        tags=("graph", "neighbourhood", "exact"),
        examples=(
            OperationExample(
                name="path_neighbourhood",
                description="Open neighbourhood of {a} in path a-b-c includes b only.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"]],
                    },
                    "selected_vertices": ["a"],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
