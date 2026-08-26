"""Public exact rational-function operation adapters and certificate checks."""

from __future__ import annotations

from jacobian.math.polynomials._conversions import (
    rational_function_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.rational_functions._models import (
    HermiteReductionRequest,
    HermiteReductionResult,
)
from jacobian.math.polynomials.rational_functions.operations import hermite_reduction


def compute_hermite_reduction(
    request: HermiteReductionRequest,
) -> HermiteReductionResult:
    rational_part, remainder = hermite_reduction(request.function)
    return HermiteReductionResult._from_kernel(
        function=request.function,
        rational_part=rational_part,
        remainder=remainder,
    )


def verify_hermite_reduction_result(result: HermiteReductionResult) -> bool:
    """Check an independently supplied Hermite certificate in its admitted envelope.

    This verifier is deliberately separate from result parsing.  The request
    and canonical result carriers bound every polynomial passed to SymPy, while
    this direct identity check establishes the defining reduction invariant.
    """

    from sympy import Poly, cancel, diff, fraction

    try:
        (variable,) = symbols_for_variables(result.function.variables)
        source = rational_function_to_sympy(result.function)
        rational_part = rational_function_to_sympy(result.rational_part)
        remainder = cancel(rational_function_to_sympy(result.remainder))
        if cancel(diff(rational_part, variable) + remainder - source) != 0:
            return False

        remainder_numerator, remainder_denominator = fraction(remainder)
        numerator = Poly(remainder_numerator, variable, domain="QQ")
        denominator = Poly(remainder_denominator, variable, domain="QQ")
        if not numerator.is_zero and numerator.degree() >= denominator.degree():
            return False
        if denominator.gcd(denominator.diff()).degree() != 0:
            return False

        part_numerator, part_denominator = fraction(cancel(rational_part))
        quotient, _ = Poly(part_numerator, variable, domain="QQ").div(
            Poly(part_denominator, variable, domain="QQ")
        )
        return bool(quotient.nth(0) == 0)
    except (TypeError, ValueError):
        return False


__all__ = ["compute_hermite_reduction", "verify_hermite_reduction_result"]
