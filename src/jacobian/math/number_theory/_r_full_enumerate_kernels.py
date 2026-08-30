"""Exact kernel for bounded r-full integer enumeration.

An integer n > 1 is r-full when every prime factor appears to exponent at least r.
We generate r-full numbers multiplicatively: start from 1, then for each prime p
with p^r <= cutoff, multiply existing family members by p^r, p^(r+1), ... as
long as the product is <= cutoff.

This avoids scanning every integer in the interval.
"""

from __future__ import annotations

from bisect import bisect_right
from heapq import merge

from sympy import integer_nthroot
from sympy.ntheory.generate import primerange


def enumerate_r_full(cutoff: int, r: int) -> list[int]:
    """Return every r-full integer in [1, cutoff] exactly once, sorted.

    Algorithm:
    1. Find all primes p with p^r <= cutoff.
    2. Start with family = {1}.
    3. For each prime p, generate all valid powers p^r, p^(r+1), p^(r+2), ...
       up to cutoff.
    4. For each existing family member m and each prime p, add m * p^k for
       k >= r, as long as the product is <= cutoff.
    5. Iterate until fixpoint.
    """
    if cutoff < 1:
        return []

    # Collect primes p with p^r <= cutoff. Bounding the prime search by the
    # exact integer root avoids scanning the entire interval up to ``cutoff``.
    prime_bound, _ = integer_nthroot(cutoff, r)
    primes: list[int] = []
    for p in primerange(2, int(prime_bound) + 1):
        primes.append(int(p))

    family_set: set[int] = {1}
    sorted_family = [1]

    # For each prime, multiply into existing family members
    for prime in primes:
        # Generate prime^r, prime^(r+1), ... all <= cutoff
        powers: list[int] = []
        current = prime**r
        while current <= cutoff:
            powers.append(current)
            current *= prime
        if not powers:
            continue

        # Multiply every existing family member by each power
        new_values: set[int] = set()
        for pw in powers:
            max_member = cutoff // pw
            for member in sorted_family[: bisect_right(sorted_family, max_member)]:
                new_values.add(member * pw)
        fresh_values = sorted(value for value in new_values if value not in family_set)
        if fresh_values:
            family_set.update(fresh_values)
            sorted_family = list(merge(sorted_family, fresh_values))

    return sorted_family
