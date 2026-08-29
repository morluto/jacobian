"""Non-coprimality graph operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.catalog._examples import example
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


def ncg_operation[
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
    ncg_operation(
        "number_theory.integer_set.non_coprimality_graph.compute",
        "Construct the non-coprimality graph of a set of positive integers",
        (
            "Given a bounded finite set of distinct positive integers, construct "
            "the canonical simple conflict graph whose vertices are the supplied "
            "integers and whose edges join exactly the distinct pairs with gcd "
            "greater than one."
        ),
        NonCoprimalityGraphRequest,
        NonCoprimalityGraphResult,
        compute_non_coprimality_graph,
        "number-theory",
        "graph",
        "exact",
        examples=(
            example(
                "basic_fixture",
                "Non-coprimality graph of {2, 3, 4, 6}; integers are canonical decimal strings.",
                {"integers": ["2", "3", "4", "6"]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
