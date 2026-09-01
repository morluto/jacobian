"""Non-coprimality graph operation declarations."""

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.number_theory.non_coprimality_graph._models import (
    NonCoprimalityGraphRequest,
    NonCoprimalityGraphResult,
)
from jacobian.math.number_theory.non_coprimality_graph.operations import (
    construct_non_coprimality_graph,
)


def compute_non_coprimality_graph(
    request: NonCoprimalityGraphRequest,
) -> NonCoprimalityGraphResult:
    return construct_non_coprimality_graph(
        tuple(parse_canonical_integer(value) for value in request.integers)
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="number_theory.integer_set.non_coprimality_graph.compute",
        title="Construct the non-coprimality graph of a set of positive integers",
        description=(
            "Given a bounded finite set of distinct positive integers, construct "
            "the canonical simple conflict graph whose vertices are the supplied "
            "integers and whose edges join exactly the distinct pairs with gcd "
            "greater than one."
        ),
        request_type=NonCoprimalityGraphRequest,
        result_type=NonCoprimalityGraphResult,
        run=compute_non_coprimality_graph,
        tags=("number-theory", "graph", "exact"),
        examples=(
            OperationExample(
                name="basic_fixture",
                description="Non-coprimality graph of {2, 3, 4, 6}; integers are canonical decimal strings.",
                input={"integers": ["2", "3", "4", "6"]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
