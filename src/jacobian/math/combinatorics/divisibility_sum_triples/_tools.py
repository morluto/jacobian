"""Typed declarations for the divisibility-sum triples hypergraph operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.combinatorics.divisibility_sum_triples._models import (
    DivisibilitySumTriplesRequest,
    DivisibilitySumTriplesResult,
)
from jacobian.math.combinatorics.divisibility_sum_triples.operations import (
    construct_divisibility_sum_triples,
)


def _construct(request: DivisibilitySumTriplesRequest) -> DivisibilitySumTriplesResult:
    return construct_divisibility_sum_triples(request.lower, request.upper)


TOOLS: MathTools = (
    MathTool(
        operation_id="hypergraph.divisibility_sum_triples.construct",
        title="Construct the divisibility-sum triples hypergraph",
        description=(
            "On an integer interval [L,U], construct the 3-uniform hypergraph "
            "whose edges are exactly the increasing triples (a,b,c) with "
            "a | (b+c)."
        ),
        request_type=DivisibilitySumTriplesRequest,
        result_type=DivisibilitySumTriplesResult,
        run=_construct,
        tags=("hypergraph", "divisibility", "combinatorics", "exact"),
        examples=(
            example(
                "small_interval",
                "Divisibility-sum triples on [1,5].",
                {"lower": 1, "upper": 5},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
