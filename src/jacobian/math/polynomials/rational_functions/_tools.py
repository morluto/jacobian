"""Rational-function operation declarations."""

from __future__ import annotations

from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationExample,
)
from jacobian.math.polynomials.rational_functions import operations as native
from jacobian.math.polynomials.rational_functions._models import (
    HermiteReductionRequest,
    HermiteReductionResult,
    PartialFractionsRequest,
    PartialFractionsResult,
)
from jacobian.math.polynomials.rational_functions.structured_tools import (
    TOOLS as STRUCTURED_TOOLS,
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


def compute_partial_fractions(
    request: PartialFractionsRequest,
) -> PartialFractionsResult:
    return native.partial_fractions(request.function)


_PARTIAL_FRACTION_FUNCTION: dict[str, object] = {
    "variables": ["x"],
    "numerator": {
        "terms": [
            {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
        ]
    },
    "denominator": {
        "terms": [
            {"coefficient": {"num": "1", "den": "1"}, "exponents": [3]},
            {"coefficient": {"num": "-1", "den": "1"}, "exponents": [2]},
            {"coefficient": {"num": "-1", "den": "1"}, "exponents": [1]},
            {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
        ]
    },
}


TOOLS: MathTools = (
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
    MathTool(
        operation_id="rational_function.partial_fractions.compute",
        title="Decompose a rational function into exact partial fractions",
        description=(
            "Return the exact factor-power partial-fraction profile of a "
            "canonical univariate QQ(x) value: the unique reduced "
            "f = A + sum n_(i,k)/g_i^k decomposition with monic irreducible "
            "denominator factors g_i, the factor-power profile, the "
            "common-denominator replay of the identity, and the independently "
            "computed Hermite remainder. General inputs allow numerator degree 6, "
            "denominator degree 3 and two-digit components; polynomial inputs "
            "allow degree 63 and 128-digit components. Irreducible factors are "
            "not split into algebraic roots."
        ),
        request_type=PartialFractionsRequest,
        result_type=PartialFractionsResult,
        run=compute_partial_fractions,
        tags=("rational-function", "partial-fractions", "exact", "factorization"),
        discovery_terms=(
            "partial fraction decomposition",
            "factor-power profile",
            "repeated pole decomposition",
        ),
        examples=(
            OperationExample(
                name="simple_and_repeated_poles",
                description=(
                    "Decompose 1/((x-1)^2*(x+1)) over QQ(x). General inputs "
                    "allow numerator degree 6, denominator degree 3 and "
                    "two-digit rational coefficient components."
                ),
                input={"function": _PARTIAL_FRACTION_FUNCTION},
            ),
        ),
    ),
)

TOOLS = TOOLS + STRUCTURED_TOOLS

__all__ = ["TOOLS"]
