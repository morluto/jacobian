"""Binary code distance graph operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.codes.nonlinear.distance_graph._models import (
    BinaryCodeDistanceGraphRequest,
    BinaryCodeDistanceGraphResult,
)
from jacobian.math.combinatorics.codes.nonlinear.distance_graph.operations import (
    compute_distance_graph,
)


def compute_distance_graph_op(
    request: BinaryCodeDistanceGraphRequest,
) -> BinaryCodeDistanceGraphResult:
    return compute_distance_graph(request.source, request.target_distance)


def bcdg_operation[
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
    bcdg_operation(
        "code.binary.explicit.distance_graph.compute",
        "Construct the Hamming distance graph of a binary code",
        (
            "For a bounded explicit binary code and selected Hamming distance, "
            "return the complete indexed simple graph whose vertices are the "
            "canonical codeword indices and whose edges join exactly the "
            "distinct codeword pairs at that Hamming distance."
        ),
        BinaryCodeDistanceGraphRequest,
        BinaryCodeDistanceGraphResult,
        compute_distance_graph_op,
        "coding-theory",
        "exact",
        examples=(
            example(
                "length3_dist1",
                "Code {000, 011, 110} at distance 1.",
                {
                    "source": {
                        "length": 3,
                        "codewords": [[0, 0, 0], [0, 1, 1], [1, 1, 0]],
                    },
                    "target_distance": 1,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
