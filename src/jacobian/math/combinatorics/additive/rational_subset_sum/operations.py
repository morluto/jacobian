"""Rational subset-sum profile kernel."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, gcd
from typing import NoReturn

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.rational_subset_sum._models import (
    MAX_SEQUENCE_LENGTH,
    RationalSubsetSumResult,
    SubsetSumRow,
)

__all__ = [
    "compute_rational_subset_sum_profile",
    "verify_rational_subset_sum_profile",
]


@dataclass(frozen=True, slots=True)
class _RationalSubsetSumPlan:
    multiplicities: tuple[tuple[Fraction, int], ...]


def _reject(code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("values",), code=f"rational_subset_sum.{code}", message=message
    )


def _decimal_digits(value: int) -> int:
    magnitude = abs(value)
    if magnitude == 0:
        return 1
    estimate = magnitude.bit_length() * 30_103 // 100_000 + 1
    while estimate > 1 and magnitude < 10 ** (estimate - 1):
        estimate -= 1
    while magnitude >= 10**estimate:
        estimate += 1
    return estimate


def _admit_values(
    values: tuple[CanonicalRational, ...],
) -> _RationalSubsetSumPlan:
    if not isinstance(values, tuple) or any(
        not isinstance(value, CanonicalRational) for value in values
    ):
        _reject("invalid_values", "values must be a tuple of canonical rationals")
    n = len(values)
    if n > MAX_SEQUENCE_LENGTH:
        _reject(
            "sequence_length_bound",
            f"at most {MAX_SEQUENCE_LENGTH} values are supported",
        )
    fractions = tuple(value.as_fraction() for value in values)
    multiplicities: dict[Fraction, int] = {}
    for value in fractions:
        multiplicities[value] = multiplicities.get(value, 0) + 1
    plan = _RationalSubsetSumPlan(
        multiplicities=tuple(multiplicities.items()),
    )
    if not n:
        return plan

    nonzero = [value for value in fractions if value]
    common_denominator_overflow = False
    if nonzero:
        common_denominator = 1
        for value in nonzero:
            common_denominator = (
                common_denominator
                // gcd(common_denominator, value.denominator)
                * value.denominator
            )
            common_denominator_overflow |= (
                _decimal_digits(common_denominator) > MAX_CANONICAL_RATIONAL_DIGITS
            )
        scaled: list[int] = [
            value.numerator * (common_denominator // value.denominator)
            for value in nonzero
        ]
        positive_span = sum(value for value in scaled if value > 0)
        negative_span = sum(value for value in scaled if value < 0)
        numerator_bound = max(abs(positive_span), abs(negative_span))
        denominator_digits = _decimal_digits(common_denominator)
        numerator_digits = _decimal_digits(numerator_bound)
        growth_digits = max(denominator_digits, numerator_digits)
    else:
        growth_digits = 1
    # Equal source values can collapse many subset vectors to one row. Group
    # them before estimating the support; the product of (multiplicity + 1)
    # bounds the number of distinct sums while retaining the actual
    # exponential work bound below.
    support_upper_bound = 1
    for value, multiplicity in multiplicities.items():
        if not value:
            continue
        support_upper_bound *= multiplicity + 1
    if nonzero:
        lattice_step: int = 0
        for scaled_value in scaled:
            lattice_step = gcd(lattice_step, abs(scaled_value))
        support_span = (positive_span - negative_span) // max(lattice_step, 1)
        support_upper_bound = min(support_upper_bound, support_span + 1)
        if (
            common_denominator_overflow
            or numerator_digits > MAX_CANONICAL_RATIONAL_DIGITS
        ):
            if len(nonzero) > 2:
                _reject(
                    "rational_growth_bound",
                    "subset-sum intermediates exceed the canonical rational digit bound",
                )
            exact_sums = {Fraction(0), *nonzero, sum(nonzero, Fraction(0))}
            if any(
                max(
                    _decimal_digits(value.numerator), _decimal_digits(value.denominator)
                )
                > MAX_CANONICAL_RATIONAL_DIGITS
                for value in exact_sums
            ):
                _reject(
                    "rational_growth_bound",
                    "subset-sum intermediates exceed the canonical rational digit bound",
                )
            growth_digits = max(
                max(
                    _decimal_digits(value.numerator), _decimal_digits(value.denominator)
                )
                for value in exact_sums
            )
    if growth_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        _reject(
            "rational_growth_bound",
            "subset-sum intermediates exceed the canonical rational digit bound",
        )
    return plan


def compute_rational_subset_sum_profile(
    values: tuple[CanonicalRational, ...],
) -> RationalSubsetSumResult:
    """Return the complete profile of all subset sums of the given rationals.

    For every subset I of the source indices, compute the sum of the
    corresponding rational values. Group by canonical rational value
    and count multiplicities.
    """
    plan = _admit_values(values)
    sum_to_count = {Fraction(0): 1}
    for value, multiplicity in plan.multiplicities:
        if not value:
            factor = 1 << multiplicity
            sum_to_count = {
                subset_sum: count * factor for subset_sum, count in sum_to_count.items()
            }
            continue
        binomial_counts = tuple(
            comb(multiplicity, chosen) for chosen in range(multiplicity + 1)
        )
        shifted_counts: dict[Fraction, int] = {}
        for subset_sum, count in sum_to_count.items():
            for chosen, binomial_count in enumerate(binomial_counts):
                shifted_sum = subset_sum + chosen * value
                shifted_counts[shifted_sum] = (
                    shifted_counts.get(shifted_sum, 0) + count * binomial_count
                )
        sum_to_count = shifted_counts

    rows = [
        SubsetSumRow(
            sum_value=CanonicalRational.from_fraction(s),
            multiplicity=count,
        )
        for s, count in sorted(sum_to_count.items())
    ]
    return RationalSubsetSumResult(
        values=values,
        rows=tuple(rows),
    )


def verify_rational_subset_sum_profile(result: RationalSubsetSumResult) -> bool:
    """Verify the complete subset-sum profile against its retained sequence."""
    try:
        expected = compute_rational_subset_sum_profile(result.values)
        return expected.rows == result.rows
    except Exception:
        return False
