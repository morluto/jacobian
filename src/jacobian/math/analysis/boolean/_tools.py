"""Exact Boolean operation declarations."""

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
        spectrum=tuple(spectrum),
        variable_count=len(request.truth_table).bit_length() - 1,
    )


TOOLS = (
    MathTool(
        operation_id="boolean.fourier.walsh_transform.compute",
        title="Compute an exact Walsh-Hadamard transform from a Boolean truth table",
        description="Compute the exact Boolean Walsh spectrum of a Boolean function from its complete truth table. "
        "The spectrum is computed by applying the fast Walsh-Hadamard transform to the sign vector "
        "(-1)^f = 1 - 2f, where f is the 0/1 truth table. The truth table is indexed in natural "
        "(little-endian) order. No floating-point arithmetic is involved.",
        request_type=BooleanTruthTableRequest,
        result_type=BooleanWalshTransformResult,
        run=_walsh_hadamard_transform,
        tags=(
            "boolean",
            "walsh",
            "fourier",
            "hadamard",
            "truth-table",
            "exact-integer",
        ),
        examples=(
            OperationExample(
                name="walsh_constant_zero_one_bit",
                description="Compute the Walsh spectrum of the 1-bit constant-zero function f=[0,0].",
                input={"truth_table": [0, 0]},
            ),
            OperationExample(
                name="walsh_identity_one_bit",
                description="Compute the Walsh spectrum of the 1-bit identity function f=[0,1].",
                input={"truth_table": [0, 1]},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
