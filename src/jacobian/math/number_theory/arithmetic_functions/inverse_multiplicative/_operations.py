"""Domain functions for inverse multiplicative function operations."""

from __future__ import annotations

from sympy import isprime

from jacobian.math.number_theory.arithmetic_functions.inverse_multiplicative._models import (
    EulerPhiPowerSumRequest,
    EulerPhiPowerSumResult,
    EulerPhiPreimageCountRequest,
    EulerPhiPreimageCountResult,
    EulerPhiPreimageRequest,
    EulerPhiPreimageResult,
)


def _divisors(n: int) -> list[int]:
    """Return the divisors of ``n`` in decreasing order."""
    small: list[int] = []
    large: list[int] = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    return large + small[::-1]


def _inverse_phi(target: int) -> set[int]:
    """Return every ``n`` with ``phi(n) = target``.

    Uses the standard recursive construction: for ``n = prod p_i^a_i`` we have
    ``phi(n) = prod p_i^(a_i-1) * (p_i - 1)``.  A prime ``p`` can only divide
    ``n`` when ``(p - 1)`` divides ``target``, so the candidate primes are
    ``p = d + 1`` for each divisor ``d`` of ``target``.  Primes are consumed in
    decreasing order so that every preimage element is produced exactly once.
    """
    if target == 1:
        return {1, 2}
    candidate_primes = [d + 1 for d in _divisors(target) if isprime(d + 1)]

    def solve(remaining: int, max_index: int) -> set[int]:
        if remaining == 1:
            return {1}
        results: set[int] = set()
        for i in range(max_index):
            p = candidate_primes[i]
            contribution = p - 1
            if remaining % contribution != 0:
                continue
            reduced = remaining // contribution
            power = p
            while True:
                for m in solve(reduced, i):
                    results.add(power * m)
                if reduced % p != 0:
                    break
                reduced //= p
                power *= p
        return results

    return solve(target, len(candidate_primes))


def compute_euler_phi_preimage(
    request: EulerPhiPreimageRequest,
) -> EulerPhiPreimageResult:
    """Compute all n such that phi(n) = target."""
    preimage = sorted(_inverse_phi(request.target))
    return EulerPhiPreimageResult(
        preimage=tuple(preimage),
        count=len(preimage),
    )


def compute_euler_phi_preimage_count(
    request: EulerPhiPreimageCountRequest,
) -> EulerPhiPreimageCountResult:
    """Count the number of n such that phi(n) = target."""
    return EulerPhiPreimageCountResult(count=len(_inverse_phi(request.target)))


def compute_euler_phi_power_sum(
    request: EulerPhiPowerSumRequest,
) -> EulerPhiPowerSumResult:
    """Compute the sum of k-th powers of the preimage of phi."""
    preimage = _inverse_phi(request.target)
    return EulerPhiPowerSumResult(
        power_sum=sum(n**request.exponent for n in preimage),
        count=len(preimage),
    )
