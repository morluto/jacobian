"""Finite-dimensional algebra operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.finite_dim_algebras._models import (
    CenterRequest,
    CenterResult,
)
from jacobian.math.finite_dim_algebras.operations import center_basis


def compute_center(request: CenterRequest) -> CenterResult:
    basis = center_basis(request.algebra)
    return CenterResult(
        center_basis=basis,
        dimension=request.algebra.dimension,
        center_dimension=len(basis),
    )


def _op[RequestT: StrictModel, ResultT: StrictModel](
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


_ZERO_ALG_2 = {
    "dimension": 2,
    "field_order": 2,
    "multiplication": [
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
    ],
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "algebra.center.compute",
        "Compute the center of a finite-dimensional algebra",
        "Compute the center {z : z*a = a*z for all a} of a finite-dimensional "
        "algebra given by structure constants over a prime field.",
        CenterRequest,
        CenterResult,
        compute_center,
        "algebra",
        "center",
        "exact",
        examples=(
            example(
                "zero_algebra",
                "Center of the 2-dimensional zero algebra over F_2.",
                {"algebra": _ZERO_ALG_2},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
