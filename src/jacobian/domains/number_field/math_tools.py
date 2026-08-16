"""Number field operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.number_field import (
    NumberFieldDiscriminantResult,
    NumberFieldRequest,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.number_field.operations import (
    compute_nf_discriminant,
)
from jacobian.math_tools import MathTool


def nf_operation[RequestT: ContractModel, ResultT: ContractModel](
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


NUMBER_FIELD_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    nf_operation(
        "number_field.discriminant.compute",
        "Compute the discriminant of a number field",
        "Compute the discriminant of a number field defined by one irreducible polynomial using SymPy.",
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
