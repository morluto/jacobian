"""Periodic congruence union prefix count kernel."""

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._periodic_kernel import (
    measure_periodic_union,
    rank_periodic_union,
    require_admitted_periodic_source,
)
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionSource,
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
    try:
        plan = require_admitted_periodic_source(source)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("source",),
            code="number_theory.periodic.execution_bound",
            message=str(exc),
        ) from exc
    occupied_count = measure_periodic_union(source, plan)
    count = rank_periodic_union(source, plan, cutoff) - rank_periodic_union(
        source, plan, 0
    )

    return PeriodicUnionPrefixCountResult(
        source=source,
        cutoff=str(cutoff),
        common_period=format_canonical_integer(plan.common_period),
        occupied_count=format_canonical_integer(occupied_count),
        count=format_canonical_integer(count),
    )
