"""Periodic congruence interval count kernel."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._periodic_kernel import (
    _inclusion_exclusion_terms,
    _sparse_union,
    _union_mask,
    rank_periodic_union,
    require_admitted_periodic_source,
)
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionSource,
)
from jacobian.math.number_theory.periodic_interval_count._models import (
    MAX_PERIODIC_INTERVAL_ENDPOINT_DIGITS,
    PeriodicIntervalCountResult,
)

__all__ = ["compute_periodic_interval_count"]


def _admit_interval_result(lower: int, upper: int) -> None:
    if type(lower) is not int or type(upper) is not int:
        raise OperationDomainValidationError(
            location=("lower", "upper"),
            code="number_theory.periodic.endpoint_type",
            message="periodic interval endpoints must be exact integers",
        )
    if max(abs(lower), abs(upper)) >= 10**MAX_PERIODIC_INTERVAL_ENDPOINT_DIGITS:
        raise OperationDomainValidationError(
            location=("lower", "upper"),
            code="number_theory.periodic.result_bound",
            message=(
                "periodic interval endpoints exceed the "
                f"{MAX_PERIODIC_INTERVAL_ENDPOINT_DIGITS}-digit bound"
            ),
        )


def compute_periodic_interval_count(
    source: PeriodicCongruenceUnionSource,
    lower: int,
    upper: int,
) -> PeriodicIntervalCountResult:
    """Count integers in [lower, upper] belonging to the periodic set.

    Uses the exact one-period profile and quotient/remainder arithmetic.
    """
    _admit_interval_result(lower, upper)
    if lower > upper:
        return PeriodicIntervalCountResult(
            source=source,
            lower=lower,
            upper=upper,
            count=0,
        )

    try:
        plan = require_admitted_periodic_source(source)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("source",),
            code="number_theory.periodic.execution_bound",
            message=str(exc),
        ) from exc
    residues: tuple[int, ...] | None = None
    terms: dict[tuple[int, int], int] | None = None
    if plan.method == "PERIOD_LIFT":
        residues = tuple(
            index
            for index, occupied in enumerate(_union_mask(source, plan.common_period))
            if occupied
        )
    elif plan.method == "SPARSE_LIFT":
        residues = tuple(sorted(_sparse_union(source, plan.common_period)))
    elif plan.method == "INCLUSION_EXCLUSION":
        terms = _inclusion_exclusion_terms(source)
    count = rank_periodic_union(
        source, plan, upper, residues=residues, terms=terms
    ) - rank_periodic_union(source, plan, lower - 1, residues=residues, terms=terms)

    return PeriodicIntervalCountResult(
        source=source,
        lower=lower,
        upper=upper,
        count=count,
    )
