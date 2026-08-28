"""Root isolation operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicOrderValue
from jacobian.math.number_theory.algebraic_numbers.root_isolation._models import (
    AlgebraicCompareRequest,
    RootIsolationResult,
    UnivariatePolynomialRequest,
)
from jacobian.math.number_theory.algebraic_numbers.root_isolation._operations import (
    compute_algebraic_compare,
    compute_root_isolation,
)


def ri_operation[RequestT: StrictModel, ResultT: StrictModel](
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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    ri_operation(
        "polynomial.roots.isolate",
        "Isolate real roots of a univariate polynomial",
        "Normalize one bounded QQ polynomial to a primitive integer source and "
        "return every distinct real root with its exact multiplicity, disjoint "
        "rational isolating interval, and directly composable canonical "
        "algebraic identity.",
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
            example(
                "cubic_with_leading_nonzero",
                "Isolate the three real roots of x^3-x; the leading coefficient must be nonzero.",
                {
                    "coefficients_descending": [
                        {"num": "1", "den": "1"},
                        {"num": "0", "den": "1"},
                        {"num": "-1", "den": "1"},
                        {"num": "0", "den": "1"},
                    ]
                },
            ),
        ),
    ),
    ri_operation(
        "algebraic_number.compare",
        "Compare two algebraic numbers",
        "Decide the exact order (LT, EQ, GT) of two bounded real algebraic "
        "numbers. Each value uses its primitive irreducible integer minimal "
        "polynomial and increasing real-root index; the source-bound result "
        "returns exact rational root-isolation evidence.",
        AlgebraicCompareRequest,
        RealAlgebraicOrderValue,
        compute_algebraic_compare,
        "algebraic",
        "comparison",
        "exact",
        examples=(
            example(
                "compare_sqrt_two_and_three",
                "Compare sqrt(2) with sqrt(3) using canonical minimal "
                "polynomials and increasing real-root indices; each polynomial "
                "must be primitive, irreducible, degree at most eight, and use "
                "coefficients of at most 1,000 digits.",
                {
                    "left": {
                        "polynomial": ["1", "0", "-2"],
                        "real_root_index": 1,
                    },
                    "right": {
                        "polynomial": ["1", "0", "-3"],
                        "real_root_index": 1,
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
