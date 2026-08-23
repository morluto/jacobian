"""Exact Ramanujan sums."""

from __future__ import annotations

from typing import SupportsIndex

__all__ = ["ramanujan_sum"]


def ramanujan_sum(modulus: SupportsIndex, frequency: SupportsIndex) -> int:
    """Return the exact Ramanujan sum ``c_modulus(frequency)``.

    The zero modulus denotes the empty reduced-residue sum.  Positive moduli
    are evaluated multiplicatively from their prime-power factorization, so
    no approximate roots of unity or reduced-residue enumeration is involved.
    """

    q = modulus.__index__()
    n = frequency.__index__()
    if q < 0:
        raise ValueError("a Ramanujan-sum modulus must be nonnegative")
    if q == 0:
        return 0

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
