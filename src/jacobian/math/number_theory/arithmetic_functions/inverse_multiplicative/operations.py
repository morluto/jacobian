"""Exact inverse Euler-totient operations."""

from sympy import isprime

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.arithmetic_functions.inverse_multiplicative._models import (
    MAX_N,
    MAX_POWER_SUM_EXPONENT,
)


def _admit_target(target: int) -> None:
    if not 1 <= target <= MAX_N:
        raise OperationDomainValidationError(
            location=("target",),
            code="inverse_totient.target_out_of_range",
            message=f"target must lie between 1 and {MAX_N}",
        )


def _divisors_descending(value: int) -> list[int]:
    small: list[int] = []
    large: list[int] = []
    divisor = 1
    while divisor * divisor <= value:
        if value % divisor == 0:
            small.append(divisor)
            if divisor != value // divisor:
                large.append(value // divisor)
        divisor += 1
    return large + small[::-1]


def euler_phi_preimages(target: int) -> tuple[int, ...]:
    """Return every positive integer whose Euler totient equals ``target``."""

    _admit_target(target)
    if target == 1:
        return (1, 2)
    candidate_primes = [
        divisor + 1 for divisor in _divisors_descending(target) if isprime(divisor + 1)
    ]

    def solve(remaining: int, maximum_index: int) -> set[int]:
        if remaining == 1:
            return {1}
        results: set[int] = set()
        for index in range(maximum_index):
            prime = candidate_primes[index]
            contribution = prime - 1
            if remaining % contribution:
                continue
            reduced = remaining // contribution
            power = prime
            while True:
                results.update(power * value for value in solve(reduced, index))
                if reduced % prime:
                    break
                reduced //= prime
                power *= prime
        return results

    return tuple(sorted(solve(target, len(candidate_primes))))


def euler_phi_preimage_count(target: int) -> int:
    """Return the size of the complete inverse-totient set."""

    return len(euler_phi_preimages(target))


def euler_phi_preimage_power_profile(target: int, exponent: int) -> tuple[int, int]:
    """Return the power sum and size of the inverse-totient set."""

    if not 1 <= exponent <= MAX_POWER_SUM_EXPONENT:
        raise OperationDomainValidationError(
            location=("exponent",),
            code="inverse_totient.exponent_out_of_range",
            message=(f"exponent must lie between 1 and {MAX_POWER_SUM_EXPONENT}"),
        )
    preimage = euler_phi_preimages(target)
    return sum(value**exponent for value in preimage), len(preimage)


def euler_phi_preimage_power_sum(target: int, exponent: int) -> int:
    """Return the exact sum of ``exponent`` powers over the inverse-totient set."""

    return euler_phi_preimage_power_profile(target, exponent)[0]


__all__ = [
    "euler_phi_preimage_count",
    "euler_phi_preimage_power_profile",
    "euler_phi_preimage_power_sum",
    "euler_phi_preimages",
]
