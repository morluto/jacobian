"""Number field operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.number_fields._discriminant_process import (
    compute_nf_discriminant,
)
from jacobian.math.number_theory.number_fields._models import (
    NumberFieldDiscriminantResult,
    NumberFieldRequest,
)


def nf_operation[RequestT: StrictModel, ResultT: StrictModel](
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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    nf_operation(
        "number_field.discriminant.compute",
        "Compute the discriminant of a number field",
        "Compute the discriminant of a number field defined by one irreducible polynomial in an isolated SymPy worker, or return UNKNOWN if its bounded execution cannot establish a result.",
        NumberFieldRequest,
        NumberFieldDiscriminantResult,
        compute_nf_discriminant,
        "number-field",
        "discriminant",
        "exact",
        examples=(
            example(
                "quadratic_disc",
                "Discriminant of x^2-2.",
                {"coefficients_descending": ["1", "0", "-2"], "variable": "x"},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
