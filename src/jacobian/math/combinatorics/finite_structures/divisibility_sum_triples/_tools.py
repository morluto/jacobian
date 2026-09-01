"""Divisibility-sum triple hypergraph operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.finite_structures.divisibility_sum_triples._models import (
    DivisibilitySumTriplesRequest,
    DivisibilitySumTriplesResult,
)
from jacobian.math.combinatorics.finite_structures.divisibility_sum_triples.operations import (
    construct_divisibility_sum_triples_hypergraph,
)


def compute_divisibility_sum_triples(
    request: DivisibilitySumTriplesRequest,
) -> DivisibilitySumTriplesResult:
    return construct_divisibility_sum_triples_hypergraph(
        request.lower_bound,
        request.upper_bound,
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="hypergraph.divisibility_sum_triples.construct",
        title="Construct the divisibility-sum triple hypergraph on an integer interval",
        description=(
            "Construct the 3-uniform hypergraph whose vertices are the "
            "integers in [L, U] and whose edges are the increasing triples "
            "(a, b, c) with L <= a < b < c <= U and a dividing b + c."
        ),
        request_type=DivisibilitySumTriplesRequest,
        result_type=DivisibilitySumTriplesResult,
        run=compute_divisibility_sum_triples,
        tags=("combinatorics", "divisibility", "exact"),
        examples=(
            OperationExample(
                name="interval_1_to_4",
                description="Divisibility-sum triples on [1, 4].",
                input={"lower_bound": 1, "upper_bound": 4},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
