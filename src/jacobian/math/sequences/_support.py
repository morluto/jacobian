"""Sequence operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog.models import MathTool, OperationExample


def sequence_operation[
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
    version: str = "2",
) -> MathTool[RequestT, ResultT]:
    """Declare one sequence math tool."""

    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_type,
        result_type=result_type,
        run=operation,
        tags=tags,
        examples=examples,
    )
