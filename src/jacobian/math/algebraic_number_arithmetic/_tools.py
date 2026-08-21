"""Algebraic number arithmetic operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.algebraic_number_arithmetic._models import (
    AlgebraicArithmeticRequest,
    AlgebraicArithmeticResult,
)
from jacobian.math.algebraic_number_arithmetic._operations import (
    compute_algebraic_add,
    compute_algebraic_multiply,
)


def _an_op[RequestT: StrictModel, ResultT: StrictModel](
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


def _element(a_num: int, b_num: int, d: int) -> dict[str, object]:
    return {
        "rational_part": {"num": str(a_num), "den": "1"},
        "radical_coefficient": {"num": str(b_num), "den": "1"},
        "radicand": d,
    }


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _an_op(
        "algebraic_number.add.compute",
        "Add two elements of Q(sqrt(d))",
        "Compute the exact sum of two elements in the quadratic "
        "number field Q(sqrt(d)). Each element is represented as "
        "a + b*sqrt(d) with rational coefficients a and b.",
        AlgebraicArithmeticRequest,
        AlgebraicArithmeticResult,
        compute_algebraic_add,
        "algebraic-number",
        "quadratic-field",
        "exact",
        examples=(
            example(
                "add_sqrt2",
                "Compute (1 + sqrt(2)) + (3 + 2*sqrt(2)) in Q(sqrt(2)).",
                {
                    "left": _element(1, 1, 2),
                    "right": _element(3, 2, 2),
                },
            ),
        ),
    ),
    _an_op(
        "algebraic_number.multiply.compute",
        "Multiply two elements of Q(sqrt(d))",
        "Compute the exact product of two elements in the quadratic "
        "number field Q(sqrt(d)). Each element is represented as "
        "a + b*sqrt(d) with rational coefficients a and b.",
        AlgebraicArithmeticRequest,
        AlgebraicArithmeticResult,
        compute_algebraic_multiply,
        "algebraic-number",
        "quadratic-field",
        "exact",
        examples=(
            example(
                "multiply_sqrt2",
                "Compute (1 + sqrt(2)) * (1 - sqrt(2)) = -1 in Q(sqrt(2)).",
                {
                    "left": _element(1, 1, 2),
                    "right": _element(1, -1, 2),
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
