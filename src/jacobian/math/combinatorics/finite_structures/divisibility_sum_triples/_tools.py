"""Divisibility-sum triple hypergraph operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def dst_operation[
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
    dst_operation(
        "hypergraph.divisibility_sum_triples.construct",
        "Construct the divisibility-sum triple hypergraph on an integer interval",
        (
            "Construct the 3-uniform hypergraph whose vertices are the "
            "integers in [L, U] and whose edges are the increasing triples "
            "(a, b, c) with L <= a < b < c <= U and a dividing b + c."
        ),
        DivisibilitySumTriplesRequest,
        DivisibilitySumTriplesResult,
        compute_divisibility_sum_triples,
        "combinatorics",
        "divisibility",
        "exact",
        examples=(
            example(
                "interval_1_to_4",
                "Divisibility-sum triples on [1, 4].",
                {"lower_bound": 1, "upper_bound": 4},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
