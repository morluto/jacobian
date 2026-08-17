"""Recurrence solving operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.recurrence_solving._models import (
    ClosedFormRequest,
    ClosedFormResult,
    RecurrenceFindRequest,
    RecurrenceFindResult,
)
from jacobian.math.recurrence_solving._operations import (
    compute_closed_form,
    compute_find_recurrence,
)


def rs_operation[RequestT: StrictModel, ResultT: StrictModel](
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


RECURRENCE_SOLVING_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    rs_operation(
        "sequence.recurrence.find",
        "Find the minimal linear recurrence of a sequence",
        "Find the lowest-order non-vacuous homogeneous recurrence that exactly fits the supplied finite rational sequence, or report NO_FITTING_RECURRENCE.",
        RecurrenceFindRequest,
        RecurrenceFindResult,
        compute_find_recurrence,
        "sequence",
        "recurrence",
        "exact",
        examples=(
            example(
                "fib_find",
                "Find the recurrence of the Fibonacci sequence.",
                {"sequence": ["1", "1", "2", "3", "5", "8", "13", "21", "34", "55"]},
            ),
        ),
    ),
    rs_operation(
        "sequence.recurrence.closed_form.compute",
        "Compute the closed-form of a linear recurrence",
        "Compute a SymPy-expression closed form for a characteristic polynomial of degree at most four and exactly one initial value per degree, including repeated roots.",
        ClosedFormRequest,
        ClosedFormResult,
        compute_closed_form,
        "sequence",
        "recurrence",
        "closed-form",
        "exact",
        examples=(
            example(
                "repeated_root",
                "Solve the recurrence with characteristic polynomial (x-1)^2.",
                {
                    "characteristic_coefficients": ["1", "-2", "1"],
                    "initial_values": ["2", "5"],
                },
            ),
        ),
    ),
)

TOOLS = RECURRENCE_SOLVING_OPERATIONS

__all__ = ["TOOLS"]
