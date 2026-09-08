"""Rational-function operation declarations."""

from __future__ import annotations

from jacobian.catalog.models import (
    MathTool,
    OperationExample,
)
from jacobian.math.polynomials.rational_functions import operations as native
from jacobian.math.polynomials.rational_functions._models import (
    HermiteReductionRequest,
    HermiteReductionResult,
)


def compute_hermite_reduction(
    request: HermiteReductionRequest,
) -> HermiteReductionResult:
    rational_part, remainder = native.hermite_reduction(request.function)
    return HermiteReductionResult._from_kernel(
        function=request.function,
        rational_part=rational_part,
        remainder=remainder,
    )


TOOLS = (
    MathTool(
        operation_id="rational_function.hermite_reduction.compute",
        title="Reduce a rational function modulo exact derivatives",
        description=(
            "Return the canonical exact decomposition f = R' + H over QQ(x), "
            "where H is proper with square-free denominator. The result also "
            "completely decides whether f has a rational primitive; a nonzero "
            "H does not rule out a formal primitive involving logarithms. The "
            "current conservative envelope admits numerator degree 6, denominator "
            "degree 3 for general rational inputs; polynomial inputs allow degree 63 "
            "and 128-digit rational components subject to primitive denominator growth "
            "or two-digit components for other rational functions."
        ),
        request_type=HermiteReductionRequest,
        result_type=HermiteReductionResult,
        run=compute_hermite_reduction,
        tags=("rational-function", "Hermite-reduction", "exact", "primitive"),
        examples=(
            OperationExample(
                name="simple_and_repeated_poles",
                description="Use canonical univariate QQ(x), one variable x. General inputs: numerator degree 6, denominator degree 3, two-digit rational coefficient components. Polynomial inputs: degree 63, 128-digit components with admitted primitive denominator growth.",
                input={
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
