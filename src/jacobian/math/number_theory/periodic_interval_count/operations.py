"""Periodic congruence interval count kernel."""

from __future__ import annotations

from jacobian.math.number_theory.periodic_interval_count._models import (
    PeriodicIntervalCountResult,
)

__all__ = ["compute_periodic_interval_count"]


def compute_periodic_interval_count(
    source,
    lower: int,
    upper: int,
) -> PeriodicIntervalCountResult:
    """Count integers in [lower, upper] belonging to the periodic set.

    Uses direct enumeration over the interval.
    """
    if lower > upper:
        return PeriodicIntervalCountResult(
            source=source,
            lower=lower,
            upper=upper,
            count=0,
        )

    # Build the set of (modulus, residues) from the source
    subsets = source.subsets
    complement = source.complement

    count = 0
    for x in range(lower, upper + 1):
        in_set = False
        for subset in subsets:
            modulus = int(subset.modulus)
            residues = set(int(r) for r in subset.residues)
            if (x % modulus) in residues or (x % modulus) in residues:
                in_set = True
                break
        if complement:
            in_set = not in_set
        if in_set:
            count += 1

    return PeriodicIntervalCountResult(
        source=source,
        lower=lower,
        upper=upper,
        count=count,
    )
