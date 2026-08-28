"""Typed declarations for coalgebra operations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.coalgebras._models import (
    ComultiplicationRequest,
    ComultiplicationResult,
    GroupLikeElementsRequest,
    GroupLikeElementsResult,
)
from jacobian.math.coalgebras.operations import (
    comultiplication,
    group_like_elements,
)


def _comultiplication(request: ComultiplicationRequest) -> ComultiplicationResult:
    return ComultiplicationResult._from_kernel(
        request, comultiplication(request.coalgebra, request.element_index)
    )


def _group_like(request: GroupLikeElementsRequest) -> GroupLikeElementsResult:
    return GroupLikeElementsResult._from_kernel(
        request, elements=group_like_elements(request.coalgebra)
    )


def coalgebra_operation[
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


# A 2-dimensional coalgebra over GF(5): Delta(c_0) = c_0 (x) c_0 and
# Delta(c_1) = c_0 (x) c_1 + c_1 (x) c_0, with epsilon(c_0) = 1,
# epsilon(c_1) = 0. Entry (j, k) of slice i is the coefficient of
# c_j (x) c_k in Delta(c_i).
_EXAMPLE_COALGEBRA: dict[str, Any] = {
    "prime": 5,
    "dimension": 2,
    "comultiplication": [
        [[1, 0], [0, 0]],
        [[0, 1], [1, 0]],
    ],
    "counit": [1, 0],
}

_COMULTIPLICATION_EXAMPLE: dict[str, Any] = {
    "coalgebra": _EXAMPLE_COALGEBRA,
    "element_index": 0,
}

_GROUP_LIKE_EXAMPLE: dict[str, Any] = {
    "coalgebra": _EXAMPLE_COALGEBRA,
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    coalgebra_operation(
        "coalgebra.comultiplication.compute",
        "Compute comultiplication Delta(c_i) for a coalgebra basis element",
        "Given a finite-dimensional coalgebra over GF(p) and a basis element "
        "index, compute the comultiplication Delta(c_i) as a matrix of "
        "coefficients over GF(p). Entry (j, k) is the coefficient of "
        "c_j (x) c_k in Delta(c_i).",
        ComultiplicationRequest,
        ComultiplicationResult,
        _comultiplication,
        "coalgebra",
        "comultiplication",
        "exact",
        examples=(
            example(
                "two_dim_comultiplication",
                "Compute Delta(c_0) for a 2D coalgebra over GF(5).",
                _COMULTIPLICATION_EXAMPLE,
            ),
        ),
    ),
    coalgebra_operation(
        "coalgebra.group_like_elements.compute",
        "Find all group-like elements in a coalgebra",
        "Find every group-like element g of a finite-dimensional coalgebra "
        "over GF(p): Delta(g) = g (x) g and epsilon(g) = 1. Enumerates the "
        "whole element space and reconstructs each candidate passing the "
        "counit filter, so requests require the derived scan work to fit "
        "the documented budget. Group-like elements are linearly "
        "independent and each spans a one-dimensional subcoalgebra.",
        GroupLikeElementsRequest,
        GroupLikeElementsResult,
        _group_like,
        "coalgebra",
        "group-like",
        "exact",
        examples=(
            example(
                "two_dim_group_like",
                "Find group-like elements in a 2D coalgebra over GF(5).",
                _GROUP_LIKE_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
