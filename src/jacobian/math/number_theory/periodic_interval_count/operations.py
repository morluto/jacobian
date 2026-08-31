"""Periodic congruence interval count kernel."""

from __future__ import annotations

from jacobian.canonical import (
    CanonicalLimits,
    canonicalize_json,
    format_canonical_integer,
)
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
    PeriodicIntervalCountResult,
)

__all__ = ["compute_periodic_interval_count"]


def _integer_decimal_digit_bound(value: int) -> int:
    if value == 0:
        return 1
    return (abs(value).bit_length() * 30_103) // 100_000 + 1


def _admit_interval_result(
    source: PeriodicCongruenceUnionSource, lower: int, upper: int
) -> None:
    lower_digits = _integer_decimal_digit_bound(lower)
    upper_digits = _integer_decimal_digit_bound(upper)
    count_digits = max(lower_digits, upper_digits) + 1
    source_bytes = len(canonicalize_json(source.model_dump(mode="json")))
    predicted_bytes = source_bytes + lower_digits + upper_digits + count_digits + 512
    if predicted_bytes > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("lower", "upper"),
            code="number_theory.periodic.result_bound",
            message="periodic interval endpoints and count exceed the output budget",
        )


def compute_periodic_interval_count(
    source: PeriodicCongruenceUnionSource,
    lower: int,
    upper: int,
) -> PeriodicIntervalCountResult:
    """Count integers in [lower, upper] belonging to the periodic set.

    Uses the exact one-period profile and quotient/remainder arithmetic.
    """
    _admit_interval_result(source, lower, upper)
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
    residues: tuple[int, ...] | None = None
    terms: dict[tuple[int, int], int] | None = None
    if plan.method == "PERIOD_LIFT":
        residues = tuple(
            index for index, occupied in enumerate(_union_mask(source, plan.common_period)) if occupied
        )
    elif plan.method == "SPARSE_LIFT":
        residues = tuple(sorted(_sparse_union(source, plan.common_period)))
    elif plan.method == "INCLUSION_EXCLUSION":
        terms = _inclusion_exclusion_terms(source)
    count = rank_periodic_union(
        source, plan, upper, residues=residues, terms=terms
    ) - rank_periodic_union(
        source, plan, lower - 1, residues=residues, terms=terms
    )

    return PeriodicIntervalCountResult(
        source=source,
        lower=format_canonical_integer(lower),
        upper=format_canonical_integer(upper),
        count=format_canonical_integer(count),
    )
