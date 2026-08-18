"""Finite-dimensional algebra operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.finite_dim_algebras._models import (
    CenterRequest,
    CenterResult,
    RadicalRequest,
    RadicalResult,
)
from jacobian.math.finite_dim_algebras._operations import (
    compute_center,
    compute_radical,
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
                "commutative_algebra",
                "Center of a 2D commutative algebra over F_2.",
                {
                    "algebra": {
                        "dimension": 2,
                        "field_order": 2,
                        "multiplication": [[0, 0], [0, 0]],
                    },
                },
            ),
        ),
    ),
    _op(
        "algebra.radical.compute",
        "Compute the Jacobson radical of a finite-dimensional algebra",
        "Compute the Jacobson radical of a finite-dimensional algebra given "
        "by structure constants over a prime field.",
        RadicalRequest,
        RadicalResult,
        compute_radical,
        "algebra",
        "radical",
        "exact",
        examples=(
            example(
                "semisimple_algebra",
                "Jacobson radical of a semisimple algebra.",
                {
                    "algebra": {
                        "dimension": 2,
                        "field_order": 2,
                        "multiplication": [[0, 0], [0, 0]],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
