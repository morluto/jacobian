"""Exact Boolean operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.analysis.boolean._models import (
    BooleanTruthTableRequest,
    BooleanWalshTransformResult,
)
from jacobian.math.analysis.boolean.operations import walsh_hadamard_transform


def _walsh_hadamard_transform(
    request: BooleanTruthTableRequest,
) -> BooleanWalshTransformResult:
    spectrum = walsh_hadamard_transform(request.truth_table)
    return BooleanWalshTransformResult(
        spectrum=tuple(format_canonical_integer(value) for value in spectrum),
        variable_count=len(request.truth_table).bit_length() - 1,
    )


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


TOOLS = (
    boolean_operation(
        "boolean.fourier.walsh_transform.compute",
        "Compute an exact Walsh-Hadamard transform from a Boolean truth table",
        "Compute the exact Boolean Walsh spectrum of a Boolean function from its complete truth table. "
        "The spectrum is computed by applying the fast Walsh-Hadamard transform to the sign vector "
        "(-1)^f = 1 - 2f, where f is the 0/1 truth table. The truth table is indexed in natural "
        "(little-endian) order. No floating-point arithmetic is involved.",
        BooleanTruthTableRequest,
        BooleanWalshTransformResult,
        _walsh_hadamard_transform,
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
    ),
)


__all__ = ["TOOLS"]
