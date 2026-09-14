"""Typed polynomial-expression normalization declaration."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials._expression_normalize import (
    PolynomialExpressionNormalizeRequest,
    PolynomialExpressionNormalizeResult,
    PolynomialExpressionSource,
    normalize_polynomial_expression,
)


def _run(
    request: PolynomialExpressionNormalizeRequest,
) -> PolynomialExpressionNormalizeResult:
    return normalize_polynomial_expression(
        PolynomialExpressionSource(
            coefficient_domain=request.coefficient_domain,
            variables=request.variables,
            expression=request.expression,
        )
    )


POLYNOMIAL_EXPRESSION_NORMALIZE_OPERATION = MathTool(
    operation_id="polynomial.expression.normalize",
    title="Normalize a typed polynomial expression",
    description=(
        "Expand a bounded typed non-evaluating AST containing only exact literals, "
        "declared variables, finite addition, finite multiplication, and "
        "nonnegative powers into a canonical rational polynomial; the supplied "
        "variable axis is ordered and every literal must belong to the declared "
        "coefficient domain."
    ),
    request_type=PolynomialExpressionNormalizeRequest,
    result_type=PolynomialExpressionNormalizeResult,
    run=_run,
    tags=("polynomial", "expression", "normalization", "exact"),
    examples=(
        OperationExample(
            name="square_binomial",
            description=(
                "Normalize (x + 1)^2 over ZZ into its canonical QQ embedding; "
                "the AST must use only the declared non-evaluating node grammar."
            ),
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
