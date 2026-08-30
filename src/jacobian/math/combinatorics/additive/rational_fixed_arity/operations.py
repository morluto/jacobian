"""Rational fixed-arity sum profile kernel."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import gcd

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.rational_fixed_arity._models import (
    RationalFixedAritySumResult,
    SumProfileRow,
)

__all__ = ["compute_rational_fixed_arity_sum_profile"]

MAX_ENUMERATION_WORK = 20_000_000
MAX_RESULT_BYTES = CanonicalLimits().max_output_bytes
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1


@dataclass(frozen=True, slots=True)
class _AdmissionPlan:
    fractions: tuple[Fraction, ...]
    candidate_count: int


def _reject(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(location=location, code=code, message=message)


def _common_denominator_digits(fractions: tuple[Fraction, ...]) -> int:
    common_denominator = 1
    for fraction in fractions:
        common_denominator = (
            common_denominator
            // gcd(common_denominator, fraction.denominator)
            * fraction.denominator
        )
        digits = len(format_canonical_integer(common_denominator))
        if digits > MAX_CANONICAL_RATIONAL_DIGITS:
            return digits
    return len(format_canonical_integer(common_denominator))


def _single_sum_digit_bounds(fractions: tuple[Fraction, ...]) -> tuple[int, int]:
    total = sum(fractions, Fraction(0))
    return (
        len(format_canonical_integer(total.numerator)),
        len(format_canonical_integer(total.denominator)),
    )


def _support_bound(
    fractions: tuple[Fraction, ...], arity: int, candidate_count: int
) -> tuple[int, int, int]:
    if candidate_count <= 1:
        return candidate_count, 0, 0
    multiplicities: dict[Fraction, int] = {}
    for fraction in fractions:
        multiplicities[fraction] = multiplicities.get(fraction, 0) + 1
    transition_bound = (
        len(multiplicities) * (arity + 1) * (max(multiplicities.values()) + 1)
    )
    if transition_bound > MAX_ENUMERATION_WORK:
        return candidate_count, 0, 0
    count_vectors = [0] * (arity + 1)
    count_vectors[0] = 1
    transition_work = 0
    for multiplicity in multiplicities.values():
        next_counts = [0] * (arity + 1)
        for used, count in enumerate(count_vectors):
            if not count:
                continue
            takes = min(multiplicity, arity - used) + 1
            transition_work += takes
            for take in range(takes):
                next_counts[used + take] += count
        count_vectors = next_counts
    count_vector_bound = count_vectors[arity]
    denominators = {value.denominator for value in fractions}
    common_denominator = 1
    for denominator in denominators:
        common_denominator = (
            common_denominator // gcd(common_denominator, denominator) * denominator
        )
        if (
            len(format_canonical_integer(common_denominator))
            > MAX_CANONICAL_RATIONAL_DIGITS
        ):
            break
    else:
        projection_work = len(fractions) * len(
            format_canonical_integer(common_denominator)
        )
        if projection_work > MAX_ENUMERATION_WORK:
            return count_vector_bound, transition_work, projection_work
        minimum = maximum = None
        lattice_step = 0
        for value in fractions:
            scaled = value.numerator * (common_denominator // value.denominator)
            if minimum is None:
                minimum = maximum = scaled
            else:
                assert maximum is not None
                lattice_step = gcd(lattice_step, abs(scaled - minimum))
                minimum = min(minimum, scaled)
                maximum = max(maximum, scaled)
        assert minimum is not None and maximum is not None
        if lattice_step:
            span_bound = (arity * (maximum - minimum)) // lattice_step + 1
            return (
                min(count_vector_bound, span_bound, candidate_count),
                transition_work,
                projection_work,
            )
    return count_vector_bound, transition_work, 0


def _capped_combination(n: int, k: int, cap: int) -> int:
    k = min(k, n - k)
    result = 1
    for index in range(1, k + 1):
        result = result * (n - k + index) // index
        if result > cap:
            return cap + 1
    return result


def _admit(  # noqa: C901
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
    if arity < 0:
        _reject(
            ("arity",),
            "rational_fixed_arity.arity_domain",
            "arity must be nonnegative",
        )
    if arity > MAX_SAFE_JSON_INTEGER:
        _reject(
            ("arity",),
            "rational_fixed_arity.arity_json_range",
            "arity must fit the canonical JSON integer range",
        )
    source_size = len(values)
    candidate_count = (
        _capped_combination(source_size, arity, MAX_ENUMERATION_WORK)
        if arity <= source_size
        else 0
    )
    arithmetic_digits = max(
        (max(len(value.num.lstrip("-")), len(value.den)) for value in values),
        default=1,
    )
    work = candidate_count * max(arity, 1) * arithmetic_digits
    if candidate_count > MAX_ENUMERATION_WORK or work > MAX_ENUMERATION_WORK:
        _reject(
            ("values",),
            "rational_fixed_arity.work_bound",
            "fixed-arity enumeration exceeds the admitted work bound",
        )
    source_size_estimate = 64 + sum(
        len(value.num) + len(value.den) + 24 for value in values
    )
    if source_size_estimate > MAX_RESULT_BYTES:
        _reject(
            ("values",),
            "rational_fixed_arity.result_bound",
            "the rational source exceeds the canonical output bound",
        )
    fractions = tuple(value.as_fraction() for value in values)

    maximum_numerator_digits = max(
        (len(value.num.lstrip("-")) for value in values),
        default=1,
    )
    maximum_denominator_digits = max(
        (len(value.den) for value in values if value.den != "1"),
        default=0,
    )
    shared_denominator = len({value.den for value in values}) == 1 if values else True
    common_denominator_digits = _common_denominator_digits(fractions)
    if candidate_count == 1:
        sum_numerator_digits, sum_denominator_digits = _single_sum_digit_bounds(
            fractions
        )
    elif candidate_count:
        # The sole empty sum is exactly 0/1, independent of source widths.
        if arity == 0:
            sum_numerator_digits = sum_denominator_digits = 1
        elif shared_denominator:
            # With one common denominator, only numerators add; the reduced
            # result cannot retain a denominator wider than that source.
            sum_numerator_digits = maximum_numerator_digits + (
                len(str(arity)) if arity > 1 else 0
            )
            sum_denominator_digits = max(1, maximum_denominator_digits)
        elif common_denominator_digits <= MAX_CANONICAL_RATIONAL_DIGITS:
            # Clearing all reduced denominators once is a tighter bound than
            # multiplying the widest denominator by the arity.  The scaled
            # numerator bound is intentionally conservative; the exact sum is
            # still checked by the canonical result envelope below.
            sum_numerator_digits = (
                maximum_numerator_digits
                + common_denominator_digits
                + (len(str(arity)) if arity > 1 else 0)
            )
            sum_denominator_digits = max(1, common_denominator_digits)
        else:
            sum_numerator_digits = sum_denominator_digits = (
                MAX_CANONICAL_RATIONAL_DIGITS + 1
            )
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
    support_bound, support_presolve_work, projection_work = _support_bound(
        fractions, arity, candidate_count
    )
    if work + support_presolve_work + projection_work > MAX_ENUMERATION_WORK:
        _reject(
            ("values",),
            "rational_fixed_arity.work_bound",
            "fixed-arity enumeration and support presolve exceed the admitted work bound",
        )
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
        fractions=fractions,
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
