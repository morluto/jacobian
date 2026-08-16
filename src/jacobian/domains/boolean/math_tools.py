"""Exact Boolean operation declarations."""

from collections.abc import Callable

from jacobian.contracts.base import ContractModel
from jacobian.contracts.boolean import (
    BooleanTruthTableRequest,
    BooleanWalshTransformResult,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.boolean.operations import compute_walsh_hadamard_transform
from jacobian.math_tools import MathTool


def boolean_operation[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
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


BOOLEAN_OPERATIONS = (
    boolean_operation(
        "boolean.fourier.walsh_transform.compute",
        "Compute an exact Walsh-Hadamard transform from a Boolean truth table",
        "Compute the exact integer Walsh-Hadamard spectrum of a Boolean function from its complete truth table using SymPy's fast Walsh-Hadamard transform (fwht). No floating-point arithmetic is involved.",
        BooleanTruthTableRequest,
        BooleanWalshTransformResult,
        compute_walsh_hadamard_transform,
        "boolean",
        "walsh",
        "fourier",
        "hadamard",
        "truth-table",
        "exact-integer",
        examples=(
            example(
                "walsh_majority_of_one_bit",
                "Compute the Walsh spectrum of the 1-bit Boolean function f(0)=1, f(1)=0.",
                {"truth_table": [1, 0]},
            ),
        ),
    ),
)
