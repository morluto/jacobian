"""Periodic congruence interval count kernel."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionSource,
)
from jacobian.math.number_theory.operations import periodic_congruence_union_profile
from jacobian.math.number_theory.periodic_interval_count._models import (
    PeriodicIntervalCountResult,
)

__all__ = ["compute_periodic_interval_count"]


def compute_periodic_interval_count(
    source: PeriodicCongruenceUnionSource,
    lower: int,
    upper: int,
) -> PeriodicIntervalCountResult:
    """Count integers in [lower, upper] belonging to the periodic set.

    Uses the exact one-period profile and quotient/remainder arithmetic.
    """
    if lower > upper:
        return PeriodicIntervalCountResult(
            source=source,
            lower=format_canonical_integer(lower),
            upper=format_canonical_integer(upper),
            count="0",
        )

    profile = periodic_congruence_union_profile(source)
    period = parse_canonical_integer(profile.common_period)
    occupied = tuple(
        parse_canonical_integer(value) for value in profile.occupied_residues
    )

    def rank(value: int) -> int:
        quotient, remainder = divmod(value, period)
        return quotient * len(occupied) + sum(
            residue <= remainder for residue in occupied
        )

    count = rank(upper) - rank(lower - 1)

    return PeriodicIntervalCountResult(
        source=source,
        lower=format_canonical_integer(lower),
        upper=format_canonical_integer(upper),
        count=format_canonical_integer(count),
    )
