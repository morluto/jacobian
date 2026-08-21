"""Typed declarations for linear matroid operations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.matroids._models import (
    MatroidClosureRequest,
    MatroidClosureResult,
    MatroidRankRequest,
    MatroidRankResult,
)
from jacobian.math.matroids._operations import (
    compute_closure,
    compute_rank,
)


def matroid_operation[
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
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


_RANK_EXAMPLE: dict[str, Any] = {
    "matroid": {
        "prime": 5,
        "num_rows": 2,
        "columns": [[1, 0], [0, 1], [1, 1]],
    },
}

_CLOSURE_EXAMPLE: dict[str, Any] = {
    "matroid": {
        "prime": 5,
        "num_rows": 2,
        "columns": [[1, 0], [0, 1], [1, 1]],
    },
    "subset": [0, 1],
}


MATROID_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    matroid_operation(
        "matroid.rank.compute",
        "Compute the rank of a linear matroid over a prime field",
        "Compute the rank (dimension of the column space) of a matroid "
        "represented by columns of a matrix over a prime field GF(p), using "
        "exact Gaussian elimination.",
        MatroidRankRequest,
        MatroidRankResult,
        compute_rank,
        "matroid",
        "rank",
        "linear-algebra",
        "exact",
        examples=(
            example(
                "uniform_matroid_rank_2",
                "Compute the rank of a 2x3 matrix with independent columns.",
                _RANK_EXAMPLE,
            ),
        ),
    ),
    matroid_operation(
        "matroid.closure.compute",
        "Compute the closure of a subset in a linear matroid",
        "Compute the closure (smallest flat) of a subset S in a matroid "
        "represented by columns over GF(p). The closure adds all elements "
        "that lie in the span of S.",
        MatroidClosureRequest,
        MatroidClosureResult,
        compute_closure,
        "matroid",
        "closure",
        "flat",
        "exact",
        examples=(
            example(
                "closure_of_basis",
                "Compute the closure of {0, 1} in a rank-2 matroid.",
                _CLOSURE_EXAMPLE,
            ),
        ),
    ),
)

TOOLS = MATROID_OPERATIONS

__all__ = ["TOOLS"]
