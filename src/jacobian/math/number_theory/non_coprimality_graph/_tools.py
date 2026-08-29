"""Operation declarations for the non-coprimality conflict graph."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.number_theory.non_coprimality_graph._models import (
    NonCoprimalityGraphRequest,
    NonCoprimalityGraphResult,
)
from jacobian.math.number_theory.non_coprimality_graph.operations import (
    non_coprimality_graph,
)


def _compute(request: NonCoprimalityGraphRequest) -> NonCoprimalityGraphResult:
    return non_coprimality_graph(request.elements.elements)


TOOLS: MathTools = (
    MathTool(
        operation_id="number_theory.integer_set.non_coprimality_graph.compute",
        title="Non-coprimality conflict graph",
        description=(
            "Construct the canonical simple conflict graph for a bounded "
            "finite set of positive integers. Vertices are the supplied "
            "integers; edges join exactly the distinct pairs whose greatest "
            "common divisor exceeds one. Exact computation via integer gcd."
        ),
        request_type=NonCoprimalityGraphRequest,
        result_type=NonCoprimalityGraphResult,
        run=_compute,
        tags=(
            "number-theory",
            "gcd",
            "coprimality",
            "graph",
            "exact",
        ),
        discovery_terms=(
            "non-coprimality graph",
            "coprimality conflict graph",
            "gcd graph",
            "shared factor graph",
        ),
        examples=(
            example(
                "pairwise_non_coprime",
                "All three integers share the factor 2, so the graph is K3.",
                {"elements": {"elements": ["2", "4", "6"]}},
            ),
            example(
                "pairwise_coprime",
                "Three pairwise coprime integers give an edgeless graph.",
                {"elements": {"elements": ["2", "3", "5"]}},
            ),
            example(
                "mixed_pair",
                "Edges join (2,6), (2,10), (3,6), and (6,10); (2,3) and (3,10) are coprime.",
                {"elements": {"elements": ["2", "3", "6", "10"]}},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
