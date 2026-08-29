"""Word-cube combinatorial-line hypergraph operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.word_cubes._models import (
    CombinatorialLineHypergraphRequest,
    CombinatorialLineHypergraphResult,
)
from jacobian.math.combinatorics.word_cubes.operations import (
    construct_combinatorial_line_hypergraph,
)


def compute_combinatorial_line_hypergraph(
    request: CombinatorialLineHypergraphRequest,
) -> CombinatorialLineHypergraphResult:
    return construct_combinatorial_line_hypergraph(
        request.alphabet_size, request.dimension
    )


def wc_operation[
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
    wc_operation(
        "words.combinatorial_line_hypergraph.compute",
        "Construct the combinatorial-line hypergraph of a word cube",
        (
            "Construct the complete canonical q-uniform hypergraph of "
            "combinatorial lines in the word cube [q]^d. Vertices are all "
            "length-d words over the alphabet {0,...,q-1}, and edges "
            "correspond to patterns with at least one wildcard coordinate."
        ),
        CombinatorialLineHypergraphRequest,
        CombinatorialLineHypergraphResult,
        compute_combinatorial_line_hypergraph,
        "combinatorics",
        "words",
        "exact",
        examples=(
            example(
                "binary_cube_dim2",
                "The word cube [2]^2 has 4 vertices and 5 combinatorial lines.",
                {"alphabet_size": 2, "dimension": 2},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
