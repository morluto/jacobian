"""Rational-function operation declarations."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.math.polynomials.rational_functions import operations as native
from jacobian.math.polynomials.rational_functions._models import (
    HermiteReductionRequest,
    HermiteReductionResult,
    require_hermite_reduction_budget,
)


def compute_hermite_reduction(
    request: HermiteReductionRequest,
) -> HermiteReductionResult:
    try:
        require_hermite_reduction_budget(request.function)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=(), code=exc.type, message=exc.message()
        ) from exc
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=(), code="polynomial.rational_function_admission", message=str(exc)
        ) from exc
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
