"""Root isolation operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.algebraic_numbers.real import RealAlgebraicOrderValue
from jacobian.math.number_theory.algebraic_numbers.root_isolation._models import (
    AlgebraicCompareRequest,
    RootIsolationResult,
    UnivariatePolynomialRequest,
)
from jacobian.math.number_theory.algebraic_numbers.root_isolation._sympy import (
    compute_algebraic_compare,
    compute_root_isolation,
)

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial.roots.isolate",
        title="Isolate real roots of a univariate polynomial",
        description="Normalize one bounded QQ polynomial to a primitive integer source and "
        "return every distinct real root with its exact multiplicity, disjoint "
        "rational isolating interval, and directly composable canonical "
        "algebraic identity.",
        request_type=UnivariatePolynomialRequest,
        result_type=RootIsolationResult,
        run=compute_root_isolation,
        tags=("polynomial", "roots", "isolation", "exact"),
        examples=(
            OperationExample(
                name="quadratic_roots",
                description="Roots of x^2-2.",
                input={
                    "polynomial": {
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2],
                                },
                                {
                                    "coefficient": {"num": "-2", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    }
                },
            ),
            OperationExample(
                name="cubic_with_leading_nonzero",
                description="Isolate the three real roots of x^3-x; the leading coefficient must be nonzero.",
                input={
                    "polynomial": {
                        "variables": ["x"],
                        "polynomial": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [3],
                                },
                                {
                                    "coefficient": {"num": "-1", "den": "1"},
                                    "exponents": [1],
                                },
                            ]
                        },
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="algebraic_number.compare",
        title="Compare two algebraic numbers",
        description="Decide the exact order (LT, EQ, GT) of two bounded real algebraic "
        "numbers. Each value uses its primitive irreducible integer minimal "
        "polynomial and increasing real-root index; the source-bound result "
        "returns exact rational root-isolation evidence.",
        request_type=AlgebraicCompareRequest,
        result_type=RealAlgebraicOrderValue,
        run=compute_algebraic_compare,
        tags=("algebraic", "comparison", "exact"),
        examples=(
            OperationExample(
                name="compare_sqrt_two_and_three",
                description="Compare sqrt(2) with sqrt(3) using canonical minimal "
                "polynomials and increasing real-root indices; each polynomial "
                "must be primitive, irreducible, degree at most eight, and use "
                "coefficients of at most 1,000 digits.",
                input={
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
