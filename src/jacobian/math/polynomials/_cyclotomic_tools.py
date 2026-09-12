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
        "Compute Phi_n with SymPy's maintained exact ZZ polynomial kernel and "
        "return its source index and Euler totient degree."
    ),
    request_type=CyclotomicRequest,
    result_type=CyclotomicResult,
    run=cyclotomic,
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
