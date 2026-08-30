"""Open-neighbourhood operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def neighborhood_operation[
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


TOOLS: MathTools = (
    neighborhood_operation(
        "graph.neighborhood.open.compute",
        "Compute the open neighbourhood of a selected vertex set",
        (
            "For a finite simple undirected graph G and a selected vertex set S, "
            "return the exact sorted open neighbourhood N_G(S) consisting of all "
            "vertices outside S adjacent to at least one member of S, in canonical "
            "source-vertex order."
        ),
        NeighborhoodRequest,
        NeighborhoodResult,
        compute_open_neighborhood,
        "graph",
        "neighbourhood",
        "exact",
        examples=(
            example(
                "path_neighbourhood",
                "Open neighbourhood of {a} in path a-b-c includes b only.",
                {
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
