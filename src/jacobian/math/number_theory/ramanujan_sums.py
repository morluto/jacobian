"""Exact Ramanujan sums."""

from __future__ import annotations

from typing import SupportsIndex

from jacobian.canonical import parse_canonical_integer
from jacobian.math.arithmetic.values import IntegerValue
from jacobian.math.number_theory._models import _MAX_INTEGER_LENGTH

__all__ = ["ramanujan_sum"]

# SymPy factors the modulus once and the frequency only participates in
# bounded modular reductions, while the exact result has absolute value at
# most the modulus.  The digit bound is therefore a pure factorization-work
# bound: it keeps synchronous factoring of hard semiprimes bounded while
# covering materially large useful moduli.
_MAX_MODULUS_DIGITS = 12


def _as_python_integer(value: SupportsIndex | IntegerValue) -> int:
    """Return one admitted integer input as its Python integer value."""

    if isinstance(value, IntegerValue):
        return parse_canonical_integer(value.value)
    return value.__index__()


def ramanujan_sum(
    modulus: SupportsIndex | IntegerValue, frequency: SupportsIndex | IntegerValue
) -> int:
    """Return the exact Ramanujan sum ``c_modulus(frequency)``.

    The zero modulus denotes the empty reduced-residue sum.  Positive moduli
    are evaluated multiplicatively from their prime-power factorization, so
    no approximate roots of unity or reduced-residue enumeration is involved.
    A positive modulus must have at most 12 decimal digits so that the
    factorization work stays bounded, and the frequency must fit the same
    256-character canonical integer envelope as the wire request, so every
    entry point performs deterministic bounded modular work.
    """

    q = _as_python_integer(modulus)
    n = _as_python_integer(frequency)
    if q < 0:
        raise ValueError("a Ramanujan-sum modulus must be nonnegative")
    # Character-length parity with the wire contract: positive frequencies
    # carry at most 256 digits and negative frequencies at most 255 plus one
    # sign character, checked via magnitudes without materializing digits.
    if n >= 10**_MAX_INTEGER_LENGTH or n <= -(10 ** (_MAX_INTEGER_LENGTH - 1)):
        raise ValueError(
            "a Ramanujan-sum frequency must carry at most "
            f"{_MAX_INTEGER_LENGTH} characters"
        )
    if q == 0:
        return 0
    if q >= 10**_MAX_MODULUS_DIGITS:
        raise ValueError(
            "a Ramanujan-sum modulus must have at most "
            f"{_MAX_MODULUS_DIGITS} decimal digits"
        )

    from sympy import factorint

    result = 1
    for prime, exponent in factorint(q).items():
        prime = int(prime)
        exponent = int(exponent)
        previous_power = prime ** (exponent - 1)
        prime_power = previous_power * prime
        if n % prime_power == 0:
            result *= previous_power * (prime - 1)
        elif n % previous_power == 0:
            result *= -previous_power
        else:
            return 0
    return result
