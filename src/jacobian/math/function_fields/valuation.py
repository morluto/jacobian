"""Exact place valuations shared by the contracts and the operations.

Both the value contracts (which re-derive a decoded result's valuations) and the
kernels (which compute them) need the same arithmetic, and a contract module may
not import its owner's operations. Keeping the valuation here lets each side use
it without the reentry that ``tools/check_architecture.py`` forbids.
"""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.function_fields._models import (
    FunctionFieldPlace,
    PrimeFieldRationalFunction,
)

__all__ = ["place_valuation_of_rational_function"]


def place_valuation_of_rational_function(
    value: PrimeFieldRationalFunction, place: FunctionFieldPlace
) -> int:
    prime = value.characteristic
    num = list(value.numerator.coefficients)
    den = list(value.denominator.coefficients)
    if place.kind == "INFINITE":
        return (len(den) - 1) - (len(num) - 1)
    place_polynomial = place.prime_polynomial
    if place_polynomial is None:
        raise OperationDomainValidationError(
            location=("place", "prime_polynomial"),
            code="function_field.finite_place_polynomial",
            message="a finite place requires its prime polynomial",
        )
    divisor = list(place_polynomial.coefficients)

    def order(poly: list[int]) -> int:
        count = 0
        while len(poly) >= len(divisor):
            quotient, remainder = _poly_divmod_local(poly, divisor, prime)
            if any(remainder):
                break
            count += 1
            poly = quotient
        return count

    return order(num) - order(den)


def _poly_divmod_local(
    dividend: list[int], divisor: list[int], prime: int
) -> tuple[list[int], list[int]]:
    dividend = [x % prime for x in dividend]
    while len(dividend) > 1 and dividend[-1] == 0:
        dividend.pop()
    divisor = [x % prime for x in divisor]
    while len(divisor) > 1 and divisor[-1] == 0:
        divisor.pop()
    if len(dividend) < len(divisor):
        return [0], dividend
    quotient = [0] * (len(dividend) - len(divisor) + 1)
    inv = pow(divisor[-1], -1, prime)
    while len(dividend) >= len(divisor) and any(dividend):
        shift = len(dividend) - len(divisor)
        factor = dividend[-1] * inv % prime
        quotient[shift] = factor
        for i, c in enumerate(divisor):
            dividend[shift + i] = (dividend[shift + i] - factor * c) % prime
        while len(dividend) > 1 and dividend[-1] == 0:
            dividend.pop()
    return quotient, dividend
