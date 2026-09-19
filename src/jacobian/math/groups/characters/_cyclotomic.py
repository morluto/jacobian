"""Exact arithmetic in Q(zeta_N) in the power basis.

Values are represented as rational coefficient tuples in ascending powers of a
fixed primitive ``order``-th root of unity, reduced modulo the cyclotomic
polynomial of that order.  Every routine is exact integer/fraction arithmetic;
no numerical approximation enters a value, comparison, or conjugation.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

MAX_ARITHMETIC_ORDER = 60
# For the supported arithmetic orders every cyclotomic polynomial coefficient
# has at most one decimal digit.  This is an admission constant as well as a
# checked backend-envelope fact: extending the order envelope must revisit it.
MAX_CYCLOTOMIC_REDUCTION_COEFFICIENT_DIGITS = 1


def euler_phi(order: int) -> int:
    """Return Euler's totient with integer-only arithmetic."""

    result, remaining, prime = order, order, 2
    while prime * prime <= remaining:
        if remaining % prime == 0:
            while remaining % prime == 0:
                remaining //= prime
            result -= result // prime
        prime += 1 if prime == 2 else 2
    if remaining > 1:
        result -= result // remaining
    return result


def _proper_divisors(order: int) -> tuple[int, ...]:
    divisors: list[int] = []
    candidate = 1
    while candidate * candidate <= order:
        if order % candidate == 0:
            if candidate != order:
                divisors.append(candidate)
            other = order // candidate
            if other != order and other != candidate:
                divisors.append(other)
        candidate += 1
    return tuple(sorted(set(divisors)))


def _poly_divmod(
    numerator: list[Fraction], denominator: list[Fraction]
) -> tuple[list[Fraction], list[Fraction]]:
    """Exact ascending-coefficient division over Q; denominator is nonzero."""

    remainder = list(numerator)
    if len(remainder) < len(denominator):
        return [], remainder
    quotient = [Fraction(0)] * (len(remainder) - len(denominator) + 1)
    lead = denominator[-1]
    for degree in range(len(remainder) - 1, len(denominator) - 2, -1):
        coefficient = remainder[degree] / lead
        if coefficient == 0:
            continue
        shift = degree - (len(denominator) - 1)
        quotient[shift] = coefficient
        for index, value in enumerate(denominator):
            remainder[shift + index] -= coefficient * value
    while len(remainder) > 1 and remainder[-1] == 0:
        remainder.pop()
    return quotient, remainder


@lru_cache(maxsize=256)
def cyclotomic_polynomial(order: int) -> tuple[Fraction, ...]:
    """Return Phi_order in ascending coefficients (monic, exact)."""

    if order == 1:
        return (Fraction(-1), Fraction(1))
    numerator = [Fraction(0)] * (order + 1)
    numerator[0] = Fraction(-1)
    numerator[order] = Fraction(1)
    for divisor in _proper_divisors(order):
        quotient, _ = _poly_divmod(numerator, list(cyclotomic_polynomial(divisor)))
        numerator = quotient
    return tuple(numerator)


def reduce_coefficients(
    order: int, work: list[Fraction] | tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    """Reduce an arbitrary power-basis vector modulo Phi_order."""

    degree_bound = euler_phi(order)
    polynomial = list(work)
    while len(polynomial) < degree_bound:
        polynomial.append(Fraction(0))
    phi = cyclotomic_polynomial(order)
    degree = len(polynomial) - 1
    while degree >= degree_bound:
        coefficient = polynomial[degree]
        if coefficient:
            base = degree - degree_bound
            for index, value in enumerate(phi):
                polynomial[base + index] -= coefficient * value
        degree -= 1
    return tuple(polynomial[index] for index in range(degree_bound))


def zero_value(order: int) -> tuple[Fraction, ...]:
    return (Fraction(0),) * euler_phi(order)


def add_values(
    order: int, left: tuple[Fraction, ...], right: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def scale_value(
    order: int, scalar: Fraction, value: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    if scalar == 0:
        return zero_value(order)
    return tuple(scalar * coefficient for coefficient in value)


def multiply_values(
    order: int, left: tuple[Fraction, ...], right: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    if not any(left) or not any(right):
        return zero_value(order)
    product = [Fraction(0)] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        if not left_value:
            continue
        for right_index, right_value in enumerate(right):
            if right_value:
                product[left_index + right_index] += left_value * right_value
    return reduce_coefficients(order, product)


def conjugate_value(order: int, value: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    """Apply the exact involution zeta_order -> zeta_order^{-1}."""

    work = [Fraction(0)] * order
    for exponent, coefficient in enumerate(value):
        if coefficient:
            work[(-exponent) % order] += coefficient
    return reduce_coefficients(order, work)


def value_from_power(order: int, exponent: int) -> tuple[Fraction, ...]:
    """Return zeta_order**exponent as a reduced power-basis value."""

    position = exponent % order
    work = [Fraction(0)] * (position + 1)
    work[position] = Fraction(1)
    return reduce_coefficients(order, work)


__all__ = [
    "MAX_ARITHMETIC_ORDER",
    "MAX_CYCLOTOMIC_REDUCTION_COEFFICIENT_DIGITS",
    "add_values",
    "conjugate_value",
    "cyclotomic_polynomial",
    "euler_phi",
    "multiply_values",
    "reduce_coefficients",
    "scale_value",
    "value_from_power",
    "zero_value",
]
