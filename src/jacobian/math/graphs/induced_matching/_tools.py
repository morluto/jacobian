"""Maximum induced matching operation declaration."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.induced_matching._models import (
    MaximumInducedMatchingRequest,
    MaximumInducedMatchingResult,
)
from jacobian.math.graphs.induced_matching.operations import maximum_induced_matching


def _compute(request: MaximumInducedMatchingRequest) -> MaximumInducedMatchingResult:
    return maximum_induced_matching(
        request.graph, resource_budget=request.resource_budget
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.induced_matching.maximum.compute",
        title="Compute a maximum induced matching",
        description="Reduce source edges to an admitted conflict graph and return an exact or bounded induced matching with its induced endpoint graph.",
        request_type=MaximumInducedMatchingRequest,
        result_type=MaximumInducedMatchingResult,
        run=_compute,
        tags=("graph", "induced-matching", "optimization", "bounded"),
        examples=(
            OperationExample(
                name="path_four",
                description="The three-edge path has induced matching number one.",
                input={
                    "graph": {
                        "vertices": ["0", "1", "2", "3"],
                        "edges": [["0", "1"], ["1", "2"], ["2", "3"]],
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
