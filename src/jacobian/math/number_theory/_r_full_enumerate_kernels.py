"""Exact kernel for bounded r-full integer enumeration.

An integer n > 1 is r-full when every prime factor appears to exponent at least r.
We generate r-full numbers multiplicatively: start from 1, then for each prime p
with p^r <= cutoff, multiply existing family members by p^r, p^(r+1), ... as
long as the product is <= cutoff.

This avoids scanning every integer in the interval.
"""

from __future__ import annotations

from jacobian.math.number_theory._r_full_enumerate_models import plan_r_full_family


def enumerate_r_full(
    cutoff: int,
    r: int,
    *,
    planned_family: tuple[int, ...] | None = None,
) -> list[int]:
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
    if planned_family is not None:
        return list(planned_family)
    if cutoff < 1:
        return []

    plan = plan_r_full_family(r, cutoff)
    if plan.exceeded:
        return []
    return list(plan.family)
