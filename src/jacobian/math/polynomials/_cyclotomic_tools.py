"""Cyclotomic polynomial declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials._cyclotomic import (
    CyclotomicRequest,
    CyclotomicResult,
    cyclotomic,
)

CYCLOTOMIC_OPERATION = MathTool(
    operation_id="polynomial.cyclotomic.compute",
    title="Compute an integer cyclotomic polynomial",
    description=(
        "Compute Phi_n with python-flint's maintained exact integer-polynomial "
        "kernel and return its source index and Euler totient degree."
    ),
    request_type=CyclotomicRequest,
    result_type=CyclotomicResult,
    run=cyclotomic,
    tags=("polynomial", "cyclotomic", "integer", "totient", "exact"),
    examples=(
        OperationExample(
            name="twelfth",
            description="Compute Phi_12(x) = x^4 - x^2 + 1.",
            input={"index": 12},
        ),
    ),
)
