"""Public rational Laurent-polynomial operation declarations."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials._laurent import (
    RationalLaurentMultiplyRequest,
    _run,
)
from jacobian.math.polynomials.values import RationalLaurentPolynomial

RATIONAL_LAURENT_MULTIPLY_OPERATION = MathTool(
    operation_id="polynomial.laurent.rational.multiply.compute",
    title="Multiply rational Laurent polynomials",
    description=(
        "Multiply two canonical sparse Laurent polynomials over QQ on the same "
        "ordered variable axis. The result combines equal signed exponent vectors "
        "exactly and omits cancelled terms while retaining the axis for zero."
    ),
    request_type=RationalLaurentMultiplyRequest,
    result_type=RationalLaurentPolynomial,
    run=_run,
    tags=("polynomial", "laurent", "rational", "multiply", "exact"),
    examples=(
        OperationExample(
            name="inverse_monomials_cancel",
            description="Multiply x + x^-1 by x - x^-1.",
            input={
                "left": {
                    "variables": ["x"],
                    "terms": [
                        {"coefficient": {"num": "1", "den": "1"}, "exponents": [1]},
                        {"coefficient": {"num": "1", "den": "1"}, "exponents": [-1]},
                    ],
                },
                "right": {
                    "variables": ["x"],
                    "terms": [
                        {"coefficient": {"num": "1", "den": "1"}, "exponents": [1]},
                        {"coefficient": {"num": "-1", "den": "1"}, "exponents": [-1]},
                    ],
                },
            },
        ),
    ),
)
