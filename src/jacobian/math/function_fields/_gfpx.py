"""Exact polynomial and rational-function arithmetic over a prime field GF(p).

Polynomials are ascending coefficient tuples with residues in ``0..p-1``;
the zero polynomial is the empty tuple.  Rational functions are reduced
``(numerator, denominator)`` pairs with a monic denominator; zero is the pair
of the zero polynomial and one.  Every routine is exact modular arithmetic.
"""

from __future__ import annotations

RF = tuple[tuple[int, ...], tuple[int, ...]]
KPoly = tuple[RF, ...]

ZERO_POLY: tuple[int, ...] = ()
ONE_POLY: tuple[int, ...] = (1,)
ZERO_RF: RF = ((), (1,))
ONE_RF: RF = ((1,), (1,))


def normalize_polynomial(
    coeffs: tuple[int, ...] | list[int], prime: int
) -> tuple[int, ...]:
    values = [value % prime for value in coeffs]
    while values and values[-1] == 0:
        values.pop()
    return tuple(values)


def poly_degree(poly: tuple[int, ...]) -> int:
    return len(poly) - 1


def poly_add(
    left: tuple[int, ...], right: tuple[int, ...], prime: int
) -> tuple[int, ...]:
    length = max(len(left), len(right))
    return normalize_polynomial(
        [
            (left[index] if index < len(left) else 0)
            + (right[index] if index < len(right) else 0)
            for index in range(length)
        ],
        prime,
    )


def poly_sub(
    left: tuple[int, ...], right: tuple[int, ...], prime: int
) -> tuple[int, ...]:
    length = max(len(left), len(right))
    return normalize_polynomial(
        [
            (left[index] if index < len(left) else 0)
            - (right[index] if index < len(right) else 0)
            for index in range(length)
        ],
        prime,
    )


def poly_mul(
    left: tuple[int, ...], right: tuple[int, ...], prime: int
) -> tuple[int, ...]:
    if not left or not right:
        return ZERO_POLY
    product = [0] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        if not left_value:
            continue
        for right_index, right_value in enumerate(right):
            product[left_index + right_index] = (
                product[left_index + right_index] + left_value * right_value
            ) % prime
    return normalize_polynomial(product, prime)


def poly_divmod(
    numerator: tuple[int, ...],
    denominator: tuple[int, ...],
    prime: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not denominator:
        raise ZeroDivisionError("polynomial division by zero")
    if len(numerator) < len(denominator):
        return ZERO_POLY, numerator
    remainder = list(numerator)
    quotient = [0] * (len(numerator) - len(denominator) + 1)
    inverse_lead = pow(denominator[-1], prime - 2, prime)
    for degree in range(len(remainder) - 1, len(denominator) - 2, -1):
        coefficient = (remainder[degree] * inverse_lead) % prime
        if coefficient:
            shift = degree - (len(denominator) - 1)
            quotient[shift] = coefficient
            for index, value in enumerate(denominator):
                remainder[shift + index] = (
                    remainder[shift + index] - coefficient * value
                ) % prime
    return (
        normalize_polynomial(quotient, prime),
        normalize_polynomial(remainder, prime),
    )


def poly_monic(poly: tuple[int, ...], prime: int) -> tuple[int, ...]:
    if not poly:
        return poly
    inverse_lead = pow(poly[-1], prime - 2, prime)
    return normalize_polynomial([value * inverse_lead for value in poly], prime)


def poly_gcd(
    left: tuple[int, ...], right: tuple[int, ...], prime: int
) -> tuple[int, ...]:
    while right:
        _, remainder = poly_divmod(left, right, prime)
        left, right = right, remainder
    return poly_monic(left, prime)


def poly_derivative(poly: tuple[int, ...], prime: int) -> tuple[int, ...]:
    return normalize_polynomial(
        [(index * value) % prime for index, value in enumerate(poly)][1:], prime
    )


def poly_powmod(
    base: tuple[int, ...],
    exponent: int,
    modulus: tuple[int, ...],
    prime: int,
) -> tuple[int, ...]:
    result = ONE_POLY
    factor = poly_divmod(base, modulus, prime)[1]
    while exponent > 0:
        if exponent & 1:
            result = poly_divmod(poly_mul(result, factor, prime), modulus, prime)[1]
        exponent >>= 1
        if exponent:
            factor = poly_divmod(poly_mul(factor, factor, prime), modulus, prime)[1]
    return result


def is_irreducible_over_gf(coeffs: tuple[int, ...], prime: int) -> bool:
    """Rabin irreducibility test for a monic polynomial over GF(prime)."""

    monic = poly_monic(coeffs, prime)
    degree = poly_degree(monic)
    if degree <= 1:
        return True
    if poly_degree(poly_gcd(monic, poly_derivative(monic, prime), prime)) > 0:
        return False
    x: tuple[int, ...] = (0, 1)
    h: tuple[int, ...] = x
    for _ in range(1, degree // 2 + 1):
        h = poly_powmod(h, prime, monic, prime)
        difference = poly_sub(h, x, prime)
        if poly_degree(poly_gcd(difference, monic, prime)) > 0:
            return False
    return True


def rf_normalize(
    numerator: tuple[int, ...],
    denominator: tuple[int, ...],
    prime: int,
) -> RF:
    numerator = normalize_polynomial(numerator, prime)
    denominator = normalize_polynomial(denominator, prime)
    if not denominator:
        raise ZeroDivisionError("rational function with zero denominator")
    if not numerator:
        return ZERO_RF
    common = poly_gcd(numerator, denominator, prime)
    if common and common != ONE_POLY:
        numerator = poly_divmod(numerator, common, prime)[0]
        denominator = poly_divmod(denominator, common, prime)[0]
    return poly_monic_rf(numerator, denominator, prime)


def poly_monic_rf(
    numerator: tuple[int, ...], denominator: tuple[int, ...], prime: int
) -> RF:
    inverse_lead = pow(denominator[-1], prime - 2, prime)
    return (
        normalize_polynomial([value * inverse_lead for value in numerator], prime),
        normalize_polynomial([value * inverse_lead for value in denominator], prime),
    )


def rf_is_zero(value: RF) -> bool:
    return not value[0]


def rf_add(left: RF, right: RF, prime: int) -> RF:
    if rf_is_zero(left):
        return right
    if rf_is_zero(right):
        return left
    numerator = poly_add(
        poly_mul(left[0], right[1], prime),
        poly_mul(right[0], left[1], prime),
        prime,
    )
    denominator = poly_mul(left[1], right[1], prime)
    return rf_normalize(numerator, denominator, prime)


def rf_sub(left: RF, right: RF, prime: int) -> RF:
    return rf_add(left, rf_neg(right, prime), prime)


def rf_neg(value: RF, prime: int) -> RF:
    return (normalize_polynomial([-c for c in value[0]], prime), value[1])


def rf_mul(left: RF, right: RF, prime: int) -> RF:
    if rf_is_zero(left) or rf_is_zero(right):
        return ZERO_RF
    return rf_normalize(
        poly_mul(left[0], right[0], prime),
        poly_mul(left[1], right[1], prime),
        prime,
    )


def rf_inv(value: RF, prime: int) -> RF:
    if rf_is_zero(value):
        raise ZeroDivisionError("inverse of the zero rational function")
    return rf_normalize(value[1], value[0], prime)


def rf_evaluate(value: RF, point: int, prime: int) -> int | None:
    """Evaluate at a field point; return None when the denominator vanishes."""

    numerator = _poly_evaluate(value[0], point, prime)
    denominator = _poly_evaluate(value[1], point, prime)
    if denominator == 0:
        return None
    return (numerator * pow(denominator, prime - 2, prime)) % prime


def _poly_evaluate(poly: tuple[int, ...], point: int, prime: int) -> int:
    result = 0
    for coefficient in reversed(poly):
        result = (result * point + coefficient) % prime
    return result


def kp_normalize(coeffs: list[RF] | tuple[RF, ...]) -> KPoly:
    values = list(coeffs)
    while values and rf_is_zero(values[-1]):
        values.pop()
    return tuple(values)


def kp_add(left: KPoly, right: KPoly, prime: int) -> KPoly:
    length = max(len(left), len(right))
    return kp_normalize(
        [
            rf_add(
                left[index] if index < len(left) else ZERO_RF,
                right[index] if index < len(right) else ZERO_RF,
                prime,
            )
            for index in range(length)
        ]
    )


def kp_sub(left: KPoly, right: KPoly, prime: int) -> KPoly:
    length = max(len(left), len(right))
    return kp_normalize(
        [
            rf_sub(
                left[index] if index < len(left) else ZERO_RF,
                right[index] if index < len(right) else ZERO_RF,
                prime,
            )
            for index in range(length)
        ]
    )


def kp_mul(left: KPoly, right: KPoly, prime: int) -> KPoly:
    if not left or not right:
        return ()
    product: list[RF] = [ZERO_RF] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        if rf_is_zero(left_value):
            continue
        for right_index, right_value in enumerate(right):
            if rf_is_zero(right_value):
                continue
            product[left_index + right_index] = rf_add(
                product[left_index + right_index],
                rf_mul(left_value, right_value, prime),
                prime,
            )
    return kp_normalize(product)


def kp_divmod(numerator: KPoly, denominator: KPoly, prime: int) -> tuple[KPoly, KPoly]:
    if not denominator:
        raise ZeroDivisionError("polynomial division by zero over GF(p)(x)")
    if len(numerator) < len(denominator):
        return (), numerator
    remainder = list(numerator)
    quotient: list[RF] = [ZERO_RF] * (len(numerator) - len(denominator) + 1)
    inverse_lead = rf_inv(denominator[-1], prime)
    for degree in range(len(remainder) - 1, len(denominator) - 2, -1):
        coefficient = rf_mul(remainder[degree], inverse_lead, prime)
        if rf_is_zero(coefficient):
            continue
        shift = degree - (len(denominator) - 1)
        quotient[shift] = coefficient
        for index, value in enumerate(denominator):
            remainder[shift + index] = rf_sub(
                remainder[shift + index], rf_mul(coefficient, value, prime), prime
            )
    return kp_normalize(quotient), kp_normalize(remainder)


def kp_gcd(left: KPoly, right: KPoly, prime: int) -> KPoly:
    while right:
        _, remainder = kp_divmod(left, right, prime)
        left, right = right, remainder
    return kp_monic(left, prime)


def kp_monic(poly: KPoly, prime: int) -> KPoly:
    if not poly:
        return poly
    inverse_lead = rf_inv(poly[-1], prime)
    return kp_normalize([rf_mul(value, inverse_lead, prime) for value in poly])


def kp_derivative(poly: KPoly, prime: int) -> KPoly:
    return kp_normalize(
        [
            rf_mul(value, ((index % prime,), ONE_POLY), prime)
            for index, value in enumerate(poly)
        ][1:]
    )


def kp_powmod(base: KPoly, exponent: int, modulus: KPoly, prime: int) -> KPoly:
    result: KPoly = (ONE_RF,)
    factor = kp_divmod(base, modulus, prime)[1]
    while exponent > 0:
        if exponent & 1:
            result = kp_divmod(kp_mul(result, factor, prime), modulus, prime)[1]
        exponent >>= 1
        if exponent:
            factor = kp_divmod(kp_mul(factor, factor, prime), modulus, prime)[1]
    return result


def kp_xgcd(left: KPoly, right: KPoly, prime: int) -> tuple[KPoly, KPoly, KPoly]:
    """Extended Euclid over GF(p)(x); returns (g, s, t) with s*left+t*right=g."""

    old_r, r = left, right
    old_s: KPoly = (ONE_RF,)
    s: KPoly = ()
    old_t: KPoly = ()
    t: KPoly = (ONE_RF,)
    while r:
        quotient, remainder = kp_divmod(old_r, r, prime)
        old_r, r = r, remainder
        old_s, s = s, kp_sub(old_s, kp_mul(quotient, s, prime), prime)
        old_t, t = t, kp_sub(old_t, kp_mul(quotient, t, prime), prime)
    if not old_r:
        return (), (), ()
    inverse_lead = rf_inv(old_r[-1], prime)
    return (
        kp_monic(old_r, prime),
        kp_normalize([rf_mul(value, inverse_lead, prime) for value in old_s]),
        kp_normalize([rf_mul(value, inverse_lead, prime) for value in old_t]),
    )


__all__ = [
    "ONE_RF",
    "RF",
    "ZERO_RF",
    "KPoly",
    "is_irreducible_over_gf",
    "kp_add",
    "kp_derivative",
    "kp_divmod",
    "kp_gcd",
    "kp_mul",
    "kp_normalize",
    "kp_powmod",
    "kp_sub",
    "kp_xgcd",
    "poly_add",
    "poly_derivative",
    "poly_divmod",
    "poly_gcd",
    "poly_monic",
    "poly_mul",
    "poly_powmod",
    "poly_sub",
    "rf_add",
    "rf_evaluate",
    "rf_inv",
    "rf_is_zero",
    "rf_mul",
    "rf_neg",
    "rf_normalize",
    "rf_sub",
]
