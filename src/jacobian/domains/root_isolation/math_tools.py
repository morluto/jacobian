"""Root isolation operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.operations import OperationExample
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


ROOT_ISOLATION_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
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
        examples=(
            example(
                "compare_sqrt_two_and_three",
                "Compare sqrt(2) with sqrt(3) from their isolating intervals.",
                {
                    "left": {
                        "polynomial": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "-2", "den": "1"},
                        ],
                        "isolating_interval_lower": {"num": "1", "den": "1"},
                        "isolating_interval_upper": {"num": "2", "den": "1"},
                    },
                    "right": {
                        "polynomial": [
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "-3", "den": "1"},
                        ],
                        "isolating_interval_lower": {"num": "1", "den": "1"},
                        "isolating_interval_upper": {"num": "2", "den": "1"},
                    },
                },
            ),
        ),
    ),
)
