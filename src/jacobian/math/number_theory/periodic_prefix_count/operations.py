"""Periodic congruence union prefix count kernel."""

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionSource,
)
from jacobian.math.number_theory.operations import (
    periodic_congruence_union_profile,
)
from jacobian.math.number_theory.periodic_prefix_count._models import (
    PeriodicUnionPrefixCountResult,
)

__all__ = ["compute_periodic_union_prefix_count"]


def compute_periodic_union_prefix_count(
    source: PeriodicCongruenceUnionSource, cutoff: int
) -> PeriodicUnionPrefixCountResult:
    """Return the exact count of integers in [1, cutoff] belonging to the periodic set.

    Uses the periodicity: if the common period is L and c residues are
    occupied in one period, then the count through X is q*c + r,
    where q = X // L, r = X % L, and c is the one-period count.
    """
    if cutoff < 0:
        raise OperationDomainValidationError(
            location=("cutoff",),
            code="periodic_prefix_count.nonnegative_cutoff",
            message="periodic prefix cutoff must be nonnegative",
        )
    profile = periodic_congruence_union_profile(source)
    period = parse_canonical_integer(profile.common_period)
    occupied = {parse_canonical_integer(r) for r in profile.occupied_residues}
    occupied_count = len(occupied)

    if period == 0:
        return PeriodicUnionPrefixCountResult(
            source=source,
            cutoff=str(cutoff),
            common_period="0",
            occupied_count=0,
            count="0",
        )

    q, r = divmod(cutoff, period)
    count = q * occupied_count
    for res in range(1, r + 1):
        if res in occupied:
            count += 1

    return PeriodicUnionPrefixCountResult(
        source=source,
        cutoff=str(cutoff),
        common_period=profile.common_period,
        occupied_count=occupied_count,
        count=format_canonical_integer(count),
    )
