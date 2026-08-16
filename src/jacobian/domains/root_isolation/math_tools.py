"""Root isolation operation declarations."""

from jacobian.contracts.base import ContractModel
from jacobian.contracts.root_isolation import (
    AlgebraicCompareRequest,
    AlgebraicCompareResult,
    RootIsolationResult,
    UnivariatePolynomialRequest,
)
from jacobian.domains._examples import example
from jacobian.domains.root_isolation.operations import (
    compute_algebraic_compare,
    compute_root_isolation,
)
from jacobian.math_tools import MathTool


def ri_operation[RequestT: ContractModel, ResultT: ContractModel](
    operation_id,
    title,
    description,
    request_model,
    result_model,
    operation,
    *tags,
    examples=(),
    version="1",
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


ROOT_ISOLATION_OPERATIONS = (
    ri_operation(
        "polynomial.roots.isolate",
        "Isolate real roots of a univariate polynomial",
        "Isolate all real roots of a univariate polynomial over QQ using SymPy exact root isolation.",
        UnivariatePolynomialRequest,
        RootIsolationResult,
        compute_root_isolation,
        "polynomial",
        "roots",
        "isolation",
        "exact",
        examples=(
            example(
                "quadratic_roots",
                "Roots of x^2-2.",
                {
                    "coefficients_descending": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "-2", "den": "1"},
                    ]
                },
            ),
        ),
    ),
    ri_operation(
        "algebraic_number.compare",
        "Compare two algebraic numbers",
        "Decide the exact order (LT, EQ, GT) of two algebraic numbers defined by minimal polynomials and isolating intervals.",
        AlgebraicCompareRequest,
        AlgebraicCompareResult,
        compute_algebraic_compare,
        "algebraic",
        "comparison",
        "exact",
        examples=(),
    ),
)
