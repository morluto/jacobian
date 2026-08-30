"""Rational fixed-arity sum profile kernel."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import comb

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.rational_fixed_arity._models import (
    MAX_ARITY,
    MAX_SEQUENCE_LENGTH,
    RationalFixedAritySumResult,
    SumProfileRow,
)

__all__ = ["compute_rational_fixed_arity_sum_profile"]

MAX_ENUMERATION_WORK = 20_000_000
MAX_RESULT_BYTES = CanonicalLimits().max_output_bytes


@dataclass(frozen=True, slots=True)
class _AdmissionPlan:
    fractions: tuple[Fraction, ...]
    candidate_count: int


def _reject(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _admit(
    values: tuple[CanonicalRational, ...],
    arity: int,
) -> _AdmissionPlan:
    """Validate source, enumeration, exact-growth, and result bounds once."""
    if type(values) is not tuple or any(
        not isinstance(value, CanonicalRational) for value in values
    ):
        _reject(
            ("values",),
            "rational_fixed_arity.values_type",
            "values must be a tuple of canonical rationals",
        )
    if type(arity) is not int:
        _reject(
            ("arity",),
            "rational_fixed_arity.arity_type",
            "arity must be an integer",
        )
    if not 0 <= arity <= MAX_ARITY:
        _reject(
            ("arity",),
            "rational_fixed_arity.arity_domain",
            f"arity must be between 0 and {MAX_ARITY}",
        )
    source_size = len(values)
    if source_size > MAX_SEQUENCE_LENGTH:
        _reject(
            ("values",),
            "rational_fixed_arity.sequence_length",
            f"values may contain at most {MAX_SEQUENCE_LENGTH} items",
        )
    candidate_count = comb(source_size, arity) if arity <= source_size else 0
    arithmetic_digits = max(
        (max(len(value.num.lstrip("-")), len(value.den)) for value in values),
        default=1,
    )
    work = candidate_count * max(arity, 1) * arithmetic_digits
    if work > MAX_ENUMERATION_WORK:
        _reject(
            ("values",),
            "rational_fixed_arity.work_bound",
            "fixed-arity enumeration exceeds the admitted work bound",
        )

    maximum_numerator_digits = max(
        (len(value.num.lstrip("-")) for value in values),
        default=1,
    )
    maximum_denominator_digits = max(
        (len(value.den) for value in values if value.den != "1"),
        default=0,
    )
    if candidate_count:
        sum_numerator_digits = (
            maximum_numerator_digits
            + max(arity - 1, 0) * maximum_denominator_digits
            + (len(str(arity)) if arity > 1 else 0)
        )
        sum_denominator_digits = max(1, arity * maximum_denominator_digits)
    else:
        sum_numerator_digits = sum_denominator_digits = 0
    if (
        candidate_count
        and max(sum_numerator_digits, sum_denominator_digits)
        > MAX_CANONICAL_RATIONAL_DIGITS
    ):
        _reject(
            ("values",),
            "rational_fixed_arity.rational_growth",
            "fixed-arity sums may exceed the canonical rational digit bound",
        )
    # Equal source values can collapse many index tuples to the same sum.  The
    # number of attainable value-count vectors is a safe support bound and is
    # much tighter for repeated inputs than the raw combination count.
    multiplicities: dict[Fraction, int] = {}
    for value in values:
        fraction = value.as_fraction()
        multiplicities[fraction] = multiplicities.get(fraction, 0) + 1
    count_vectors = [0] * (arity + 1)
    count_vectors[0] = 1
    for multiplicity in multiplicities.values():
        next_counts = [0] * (arity + 1)
        for used, count in enumerate(count_vectors):
            if not count:
                continue
            for take in range(min(multiplicity, arity - used) + 1):
                next_counts[used + take] += count
        count_vectors = next_counts
    support_bound = count_vectors[arity] if arity <= source_size else 0
    try:
        source_bytes = len(
            encode_strict_json(
                {"values": [value.model_dump(mode="json") for value in values]}
            )
        )
    except CanonicalizationError:
        _reject(
            ("values",),
            "rational_fixed_arity.result_bound",
            "the complete sum profile exceeds the canonical output bound",
        )
    row_bytes = (
        sum_numerator_digits + sum_denominator_digits + len(str(candidate_count)) + 64
    )
    result_bytes = 256 + source_bytes + support_bound * row_bytes
    if result_bytes > MAX_RESULT_BYTES:
        _reject(
            ("values",),
            "rational_fixed_arity.result_bound",
            "the exact sum profile exceeds the canonical output bound",
        )
    return _AdmissionPlan(
        fractions=tuple(value.as_fraction() for value in values),
        candidate_count=candidate_count,
    )


def compute_rational_fixed_arity_sum_profile(
    values: tuple[CanonicalRational, ...],
    arity: int,
) -> RationalFixedAritySumResult:
    """Return the complete profile of sums of exactly arity distinct indexed values.

    For each strictly increasing index h-tuple, compute the sum of the
    corresponding rational values. Group by canonical rational value and
    count multiplicities.
    """
    plan = _admit(values, arity)
    fractions = plan.fractions
    sum_to_count: dict[Fraction, int] = {}

    for indices in combinations(range(len(values)), arity):
        total = sum((fractions[i] for i in indices), Fraction(0))
        sum_to_count[total] = sum_to_count.get(total, 0) + 1

    rows = [
        SumProfileRow(
            sum_value=CanonicalRational.from_fraction(s),
            multiplicity=count,
        )
        for s, count in sorted(sum_to_count.items())
    ]
    return RationalFixedAritySumResult._from_kernel(
        values=values,
        arity=arity,
        rows=tuple(rows),
    )
