"""Exact place valuations shared by the contracts and the operations.

Both the value contracts (which re-derive a decoded result's valuations) and the
kernels (which compute them) need the same arithmetic, and a contract module may
not import its owner's operations. Keeping the valuation here lets each side use
it without the reentry that ``tools/check_architecture.py`` forbids.
"""

from __future__ import annotations

from math import isqrt

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.function_fields._gfpx import (
    is_irreducible_over_gf,
    poly_divmod,
    rf_normalize,
)
from jacobian.math.function_fields._models import (
    MAX_RIEMANN_ROCH_MEMBERSHIP_FACTOR_WORK,
    FiniteFunctionFieldElement,
    FunctionFieldDivisor,
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


def membership_profile_valuations(
    element: FiniteFunctionFieldElement, places: tuple[FunctionFieldPlace, ...]
) -> tuple[int, ...]:
    """Replay a decoded rational-field profile under one aggregate budget.

    Models have checked all polynomial and row axes already. Validate the
    relied-upon field and prime-place relations without calling an operation
    or factoring the retained element again for each row. Polynomial
    irreducibility is tested directly, once per distinct finite polynomial.
    """
    field = element.field
    prime = field.characteristic
    if any(prime % d == 0 for d in range(2, isqrt(prime) + 1)):
        raise ValueError("membership valuations require prime characteristic")
    if len(field.defining_polynomial) != 1:
        raise ValueError("membership valuations require the rational field GF(p)(x)")
    coordinate = element.coordinates[0]
    polynomials = {
        place.prime_polynomial.coefficients
        for place in places
        if place.prime_polynomial is not None
    }
    # Same cubic irreducibility unit as producer admission, plus the complete
    # bounded polynomial-division replay. Every row shares this one allowance.
    bits = prime.bit_length()
    division_work = (
        (coordinate.numerator.degree + 1) ** 2
        + (coordinate.denominator.degree + 1) ** 2
    ) * bits
    # Include one rational normalization and the support-completeness pass.
    normalization_work = (
        max(coordinate.numerator.degree, coordinate.denominator.degree) + 1
    ) ** 3 * bits
    work = (
        normalization_work
        + (len(places) + 1) * division_work
        + sum(max(1, (len(poly) - 1) ** 3) * bits for poly in polynomials)
    )
    if work > MAX_RIEMANN_ROCH_MEMBERSHIP_FACTOR_WORK:
        raise OperationResourceAdmissionError(
            location=("profile",),
            code="function_field.riemann_roch_membership_work_exceeds_envelope",
            message="aggregate decoded profile replay exceeds the admitted work bound",
        )
    request_checkpoint("before decoded membership profile replay")
    for poly in polynomials:
        request_checkpoint("during decoded membership prime-place checks")
        if not is_irreducible_over_gf(poly, prime):
            raise ValueError("membership profile place polynomial must be irreducible")
    values = []
    for place in places:
        request_checkpoint("during decoded membership valuations")
        values.append(place_valuation_of_rational_function(coordinate, place))
    return tuple(values)


def require_complete_membership_support(
    element: FiniteFunctionFieldElement,
    divisor: FunctionFieldDivisor,
    places: tuple[FunctionFieldPlace, ...],
    multiplicities: tuple[int, ...],
    valuations: tuple[int, ...],
) -> None:
    """Bind every divisor coefficient and verify no finite/infinite pole is omitted.

    Called after the aggregate valuation/irreducibility admission. Reduce the
    rational function once and divide out its listed primes; no factorization
    of the element or public-operation replay is needed.
    """
    prime = element.field.characteristic

    def key(place: FunctionFieldPlace) -> tuple[int, ...]:
        if place.prime_polynomial is None:
            return ()
        coefficients = place.prime_polynomial.coefficients
        inverse = pow(coefficients[-1], -1, prime)
        return tuple(value * inverse % prime for value in coefficients)

    source_coefficients: dict[tuple[int, ...], int] = {}
    for term in divisor.terms:
        k = key(term.place)
        if k in source_coefficients:
            raise ValueError("divisor repeats a semantic place")
        source_coefficients[k] = term.multiplicity
    keys = tuple(key(place) for place in places)
    if len(set(keys)) != len(keys):
        raise ValueError("profile repeats a semantic place")
    if not source_coefficients.keys() <= set(keys):
        raise ValueError("profile omits a divisor support place")
    for k, value in zip(keys, multiplicities, strict=True):
        if value != source_coefficients.get(k, 0):
            raise ValueError(
                "profile divisor coefficient does not match retained divisor"
            )
    coordinate = element.coordinates[0]
    request_checkpoint("before decoded membership support completeness")
    numerator, denominator = rf_normalize(
        coordinate.numerator.coefficients, coordinate.denominator.coefficients, prime
    )
    infinity = len(denominator) - len(numerator)
    if infinity and () not in keys:
        raise ValueError("profile omits the nonzero infinite valuation")
    # A zero valuation has no factor in the reduced fraction. Reuse the
    # admitted valuations so completeness only repeats successful factor
    # removal; its aggregate division bound is independent of profile size.
    for poly, valuation in zip(keys, valuations, strict=True):
        if not poly or valuation == 0:
            continue
        request_checkpoint("during decoded membership support completeness")
        if valuation > 0:
            numerator = _remove_prime_factors(numerator, poly, prime)
        else:
            denominator = _remove_prime_factors(denominator, poly, prime)
    if len(numerator) > 1 or len(denominator) > 1:
        raise ValueError("profile omits a finite zero or pole")


def _remove_prime_factors(
    value: tuple[int, ...], poly: tuple[int, ...], prime: int
) -> tuple[int, ...]:
    while len(value) >= len(poly):
        quotient, remainder = poly_divmod(value, poly, prime)
        if remainder:
            break
        value = quotient
    return value
