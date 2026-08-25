"""Bounded exact kernel for powerful-number decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from flint import fmpz
from sympy.ntheory.generate import Sieve, primerange

from jacobian.math.number_theory._powerful_models import (
    MAX_POWERFUL_CUTOFF,
    MAX_POWERFUL_INTEGER_DIGITS,
)

PowerfulConclusion = Literal[
    "POWERFUL",
    "EXPONENT_ONE",
    "ROUGH_NOT_PERFECT_POWER",
]


@dataclass(frozen=True, slots=True)
class PowerfulDecisionData:
    """Private canonical data returned by the exact decision kernel."""

    conclusion: PowerfulConclusion
    cutoff: int
    checked_through: int
    stripped_factors: tuple[tuple[int, int], ...]
    residual: int
    perfect_power: tuple[int, int] | None = None


def _ceil_fifth_root(value: int) -> int:
    root = int(fmpz(value).root(5))
    return root if root**5 == value else root + 1


def _perfect_power_witness(value: int) -> tuple[int, int] | None:
    """Return the least-prime-exponent witness for a positive integer > 1."""

    integer = fmpz(value)
    if not integer.is_perfect_power():
        return None

    # Every nontrivial power exponent has a prime divisor, and an exponent for
    # value is at most floor(log_2(value)).  FLINT supplies each exact floor
    # root; integer reconstruction decides exactness without floating point.
    for raw_exponent in primerange(2, value.bit_length() + 1):
        exponent = int(raw_exponent)
        base = int(integer.root(exponent))
        if base**exponent == value:
            return base, exponent
    raise AssertionError("FLINT perfect-power result has no exact prime exponent")


def decide_powerful_data(value: int) -> PowerfulDecisionData:
    """Decide whether every prime divisor of ``value`` has exponent at least 2.

    Let ``B = ceil(value**(1/5))``.  Small primes are stripped in ascending
    order.  An exponent-one factor is an immediate obstruction.  Otherwise the
    remaining B-rough cofactor is powerful exactly when it is 1 or a perfect
    power: a powerful non-perfect-power B-rough cofactor would have exponent
    sum at least five and therefore exceed B**5 >= value.
    """

    if not 1 <= value < 10**MAX_POWERFUL_INTEGER_DIGITS:
        raise ValueError("powerful-number value must have at most 25 digits")
    cutoff = _ceil_fifth_root(value)
    assert cutoff <= MAX_POWERFUL_CUTOFF
    residual = value
    stripped: list[tuple[int, int]] = []

    # A fresh sieve keeps prime generation request-scoped.  At the admitted
    # maximum B=100000 this enumerates exactly 9592 primes.
    for raw_prime in Sieve().primerange(2, cutoff + 1):
        prime = int(raw_prime)
        if residual % prime != 0:
            continue
        exponent = 0
        while residual % prime == 0:
            residual //= prime
            exponent += 1
        stripped.append((prime, exponent))
        if exponent == 1:
            return PowerfulDecisionData(
                conclusion="EXPONENT_ONE",
                cutoff=cutoff,
                checked_through=prime,
                stripped_factors=tuple(stripped),
                residual=residual,
            )

    if residual == 1:
        return PowerfulDecisionData(
            conclusion="POWERFUL",
            cutoff=cutoff,
            checked_through=cutoff,
            stripped_factors=tuple(stripped),
            residual=1,
        )

    witness = _perfect_power_witness(residual)
    return PowerfulDecisionData(
        conclusion=("POWERFUL" if witness is not None else "ROUGH_NOT_PERFECT_POWER"),
        cutoff=cutoff,
        checked_through=cutoff,
        stripped_factors=tuple(stripped),
        residual=residual,
        perfect_power=witness,
    )
