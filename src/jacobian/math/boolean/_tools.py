"""Exact Boolean operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.boolean._models import (
    BooleanTruthTableRequest,
    BooleanWalshTransformResult,
)
from jacobian.math.boolean._operations import compute_walsh_hadamard_transform


def boolean_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
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
        "Compute the exact Boolean Walsh spectrum of a Boolean function from its complete truth table. "
        "The spectrum is computed by applying the fast Walsh-Hadamard transform to the sign vector "
        "(-1)^f = 1 - 2f, where f is the 0/1 truth table. The truth table is indexed in natural "
        "(little-endian) order. No floating-point arithmetic is involved.",
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
                "walsh_constant_zero_one_bit",
                "Compute the Walsh spectrum of the 1-bit constant-zero function f=[0,0].",
                {"truth_table": [0, 0]},
            ),
            example(
                "walsh_identity_one_bit",
                "Compute the Walsh spectrum of the 1-bit identity function f=[0,1].",
                {"truth_table": [0, 1]},
            ),
        ),
        version="2",
    ),
)

TOOLS = BOOLEAN_OPERATIONS

__all__ = ["TOOLS"]
