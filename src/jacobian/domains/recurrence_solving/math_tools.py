"""Recurrence solving operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.operations import OperationExample
from jacobian.contracts.recurrence_solving import (
    ClosedFormRequest,
    ClosedFormResult,
    RecurrenceFindRequest,
    RecurrenceFindResult,
)
from jacobian.domains._examples import example
from jacobian.domains.recurrence_solving.operations import (
    compute_closed_form,
    compute_find_recurrence,
)
from jacobian.math_tools import MathTool


def rs_operation[RequestT: ContractModel, ResultT: ContractModel](
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
        "Find the minimal linear recurrence of a finite sequence over QQ using exact rational interpolation.",
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
        "Compute the closed-form Binet-type solution of a linear recurrence from its characteristic polynomial using SymPy rsolve.",
        ClosedFormRequest,
        ClosedFormResult,
        compute_closed_form,
        "sequence",
        "recurrence",
        "closed-form",
        "exact",
        examples=(),
    ),
)
