"""Typed polynomial-expression normalization declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials._expression_normalize import (
    PolynomialExpressionNormalizeRequest,
    PolynomialExpressionNormalizeResult,
    normalize_polynomial_expression,
)

POLYNOMIAL_EXPRESSION_NORMALIZE_OPERATION = MathTool(
    operation_id="polynomial.expression.normalize",
    title="Normalize a typed polynomial expression",
    description="Expand a bounded non-evaluating literal, variable, addition, multiplication, and nonnegative-power AST into a canonical rational polynomial.",
    request_type=PolynomialExpressionNormalizeRequest,
    result_type=PolynomialExpressionNormalizeResult,
    run=normalize_polynomial_expression,
    tags=("polynomial", "expression", "normalization", "exact"),
    examples=(
        OperationExample(
            name="square_binomial",
            description="Normalize (x + 1)^2 over ZZ into its canonical QQ embedding.",
            input={
                "coefficient_domain": "ZZ",
                "variables": ["x"],
                "expression": {
                    "kind": "POWER",
                    "base": {
                        "kind": "ADD",
                        "operands": [
                            {"kind": "VARIABLE", "name": "x"},
                            {"kind": "LITERAL", "value": {"num": "1", "den": "1"}},
                        ],
                    },
                    "exponent": 2,
                },
            },
        ),
    ),
)

__all__ = ["POLYNOMIAL_EXPRESSION_NORMALIZE_OPERATION"]
