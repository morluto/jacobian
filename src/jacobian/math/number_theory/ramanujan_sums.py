"""Exact Ramanujan sums."""

from __future__ import annotations

from typing import SupportsIndex

__all__ = ["ramanujan_sum"]

# SymPy factors the modulus once and the frequency only participates in
# bounded modular reductions, while the exact result has absolute value at
# most the modulus.  The digit bound is therefore a pure factorization-work
# bound: it keeps synchronous factoring of hard semiprimes bounded while
# covering materially large useful moduli.
_MAX_MODULUS_DIGITS = 12


def ramanujan_sum(modulus: SupportsIndex, frequency: SupportsIndex) -> int:
    """Return the exact Ramanujan sum ``c_modulus(frequency)``.

    The zero modulus denotes the empty reduced-residue sum.  Positive moduli
    are evaluated multiplicatively from their prime-power factorization, so
    no approximate roots of unity or reduced-residue enumeration is involved.
    A positive modulus must have at most 12 decimal digits so that the
    factorization work stays bounded for every entry point.
    """

    q = modulus.__index__()
    n = frequency.__index__()
    if q < 0:
        raise ValueError("a Ramanujan-sum modulus must be nonnegative")
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
