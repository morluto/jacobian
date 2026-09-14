"""Cyclotomic polynomial declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials._cyclotomic import (
    CyclotomicRequest,
    CyclotomicResult,
    _run,
)

CYCLOTOMIC_OPERATION = MathTool(
    operation_id="polynomial.cyclotomic.compute",
    title="Compute an integer cyclotomic polynomial",
    description=(
        "Compute the exact integer cyclotomic polynomial Phi_n together with "
        "its source index and Euler totient degree. The result is the minimal "
        "polynomial of a primitive n-th root of unity over the integers."
    ),
    request_type=CyclotomicRequest,
    result_type=CyclotomicResult,
    run=_run,
    tags=("polynomial", "cyclotomic", "integer", "totient", "exact"),
    discovery_terms=(
        "cyclotomic polynomial",
        "primitive roots of unity polynomial",
        "Phi_n",
    ),
    examples=(
        OperationExample(
            name="twelfth",
            description="Compute Phi_12(x) = x^4 - x^2 + 1.",
            input={"index": 12},
        ),
    ),
)
