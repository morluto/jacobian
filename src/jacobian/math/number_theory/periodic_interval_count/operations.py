"""Periodic congruence interval count kernel."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._periodic_kernel import (
    rank_periodic_union,
    require_admitted_periodic_source,
)
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionSource,
)
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

    try:
        plan = require_admitted_periodic_source(source)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("source",),
            code="number_theory.periodic.execution_bound",
            message=str(exc),
        ) from exc
    count = rank_periodic_union(source, plan, upper) - rank_periodic_union(
        source, plan, lower - 1
    )

    return PeriodicIntervalCountResult(
        source=source,
        lower=format_canonical_integer(lower),
        upper=format_canonical_integer(upper),
        count=format_canonical_integer(count),
    )
