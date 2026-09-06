"""Declarations for divisibility-incidence graph construction."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory._divisibility_graph_models import (
    DivisibilityIncidenceGraphRequest,
    DivisibilityIncidenceGraphResult,
)
from jacobian.math.number_theory.operations import divisibility_incidence_graph


def compute_divisibility_incidence_graph(
    request: DivisibilityIncidenceGraphRequest,
) -> DivisibilityIncidenceGraphResult:
    """Project a wire request into the canonical graph operation."""
    return divisibility_incidence_graph(
        request.left_family,
        request.right_family,
    )


DIVISIBILITY_GRAPH_OPERATION = MathTool(
    operation_id="number_theory.divisibility_incidence_graph.compute",
    title="Build bipartite divisibility-incidence graph for two integer families",
    description="Given left family L and right family R, build a bipartite simple graph joining l to r iff l divides r.",
    request_type=DivisibilityIncidenceGraphRequest,
    result_type=DivisibilityIncidenceGraphResult,
    run=compute_divisibility_incidence_graph,
    tags=("number-theory", "graph", "divisibility"),
    examples=(
        OperationExample(
            name="div_incidence_basic",
            description="Build divisibility graph for left={2,3} and right={6,12,5}; left and right families must have unique elements.",
            input={"left_family": ["2", "3"], "right_family": ["6", "12", "5"]},
        ),
    ),
)

__all__ = ["DIVISIBILITY_GRAPH_OPERATION"]
