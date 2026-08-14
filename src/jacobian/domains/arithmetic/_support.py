"""Arithmetic operation declarations."""

from collections.abc import Callable

from jacobian.contracts.base import ContractModel
from jacobian.contracts.operations import OperationExample
from jacobian.math_tools import MathTool


def arithmetic_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
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
    """Declare one arithmetic math tool."""

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
