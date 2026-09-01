"""Typed declarations for coalgebra operations."""

from typing import Any

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
    MathTool(
        operation_id="coalgebra.comultiplication.compute",
        title="Compute comultiplication Delta(c_i) for a coalgebra basis element",
        description="Given a finite-dimensional coalgebra over GF(p) and a basis element "
        "index, compute the comultiplication Delta(c_i) as a matrix of "
        "coefficients over GF(p). Entry (j, k) is the coefficient of "
        "c_j (x) c_k in Delta(c_i).",
        request_type=ComultiplicationRequest,
        result_type=ComultiplicationResult,
        run=_comultiplication,
        tags=("coalgebra", "comultiplication", "exact"),
        examples=(
            OperationExample(
                name="two_dim_comultiplication",
                description="Compute Delta(c_0) for a 2D coalgebra over GF(5).",
                input=_COMULTIPLICATION_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="coalgebra.group_like_elements.compute",
        title="Find all group-like elements in a coalgebra",
        description="Find every group-like element g of a finite-dimensional coalgebra "
        "over GF(p): Delta(g) = g (x) g and epsilon(g) = 1. Enumerates the "
        "whole element space and reconstructs each candidate passing the "
        "counit filter, so requests require the derived scan work to fit "
        "the documented budget. Group-like elements are linearly "
        "independent and each spans a one-dimensional subcoalgebra.",
        request_type=GroupLikeElementsRequest,
        result_type=GroupLikeElementsResult,
        run=_group_like,
        tags=("coalgebra", "group-like", "exact"),
        examples=(
            OperationExample(
                name="two_dim_group_like",
                description="Find group-like elements in a 2D coalgebra over GF(5).",
                input=_GROUP_LIKE_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
