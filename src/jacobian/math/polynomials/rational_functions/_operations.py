"""Public exact rational-function operation adapters."""

from jacobian.math.polynomials.rational_functions._models import (
    HermiteReductionRequest,
    HermiteReductionResult,
)
from jacobian.math.polynomials.rational_functions.operations import hermite_reduction


def compute_hermite_reduction(
    request: HermiteReductionRequest,
) -> HermiteReductionResult:
    rational_part, remainder = hermite_reduction(request.function)
    zero_remainder = not remainder.numerator.terms
    return HermiteReductionResult(
        function=request.function,
        rational_part=rational_part,
        remainder=remainder,
        rational_primitive_status=(
            "RATIONAL_PRIMITIVE" if zero_remainder else "NO_RATIONAL_PRIMITIVE"
        ),
        rational_primitive=rational_part if zero_remainder else None,
    )


__all__ = ["compute_hermite_reduction"]
