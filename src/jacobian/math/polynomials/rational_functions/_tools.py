"""Rational-function operation declarations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.polynomials.rational_functions._models import (
    HermiteReductionRequest,
    HermiteReductionResult,
)
from jacobian.math.polynomials.rational_functions._operations import (
    compute_hermite_reduction,
)

TOOLS = (
    MathTool(
        operation_id="rational_function.hermite_reduction.compute",
        version="1",
        title="Reduce a rational function modulo exact derivatives",
        description=(
            "Return the canonical exact decomposition f = R' + H over QQ(x), "
            "where H is proper with square-free denominator. The result also "
            "completely decides whether f has a rational primitive; a nonzero "
            "H does not rule out a formal primitive involving logarithms. The "
            "current conservative envelope admits numerator degree 6, denominator "
            "degree 3, and two-digit rational coefficient components."
        ),
        request_type=HermiteReductionRequest,
        result_type=HermiteReductionResult,
        run=compute_hermite_reduction,
        tags=("rational-function", "Hermite-reduction", "exact", "primitive"),
        examples=(
            example(
                "simple_and_repeated_poles",
                "Separate the derivative of a repeated pole from a simple-pole "
                "remainder; the function must be canonical univariate QQ(x) in "
                "one variable x, with numerator degree at most 6, denominator "
                "degree at most 3, and two-digit rational coefficient "
                "components.",
                {
                    "function": {
                        "variables": ["x"],
                        "numerator": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [0],
                                }
                            ]
                        },
                        "denominator": {
                            "terms": [
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [2],
                                },
                                {
                                    "coefficient": {"num": "-2", "den": "1"},
                                    "exponents": [1],
                                },
                                {
                                    "coefficient": {"num": "1", "den": "1"},
                                    "exponents": [0],
                                },
                            ]
                        },
                    }
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
