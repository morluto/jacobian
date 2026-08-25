"""Finite-set operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog.models import MathTool, OperationExample


def finite_set_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_type: type[RequestT],
    result_type: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    """Declare one finite-set math tool."""

    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_type,
        result_type=result_type,
        run=operation,
        tags=tags,
        examples=examples,
    )
