"""Rational subset-sum profile kernel."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import gcd
from typing import NoReturn

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
)
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.rational_subset_sum._models import (
    MAX_SEQUENCE_LENGTH,
    RationalSubsetSumResult,
    SubsetSumRow,
)

__all__ = ["compute_rational_subset_sum_profile"]

MAX_RESULT_BYTES = CanonicalLimits().max_output_bytes


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


def _admit_values(values: tuple[CanonicalRational, ...]) -> None:
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
    if not n:
        return

    nonzero = [value for value in values if value.as_fraction()]
    common_denominator_overflow = False
    if nonzero:
        fractions = [value.as_fraction() for value in nonzero]
        common_denominator = 1
        for value in fractions:
            common_denominator = (
                common_denominator
                // gcd(common_denominator, value.denominator)
                * value.denominator
            )
            common_denominator_overflow |= (
                _decimal_digits(common_denominator) > MAX_CANONICAL_RATIONAL_DIGITS
            )
        scaled = [
            value.numerator * (common_denominator // value.denominator)
            for value in fractions
        ]
        positive_span = sum(value for value in scaled if value > 0)
        negative_span = sum(value for value in scaled if value < 0)
        numerator_bound = max(abs(positive_span), abs(negative_span))
        denominator_digits = _decimal_digits(common_denominator)
        numerator_digits = _decimal_digits(numerator_bound)
        growth_digits = max(denominator_digits, numerator_digits)
    else:
        growth_digits = 1
    if growth_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        _reject(
            "rational_growth_bound",
            "subset-sum intermediates exceed the canonical rational digit bound",
        )

    source_bytes = len(
        encode_strict_json({"values": [v.model_dump(mode="json") for v in values]})
    )
    # Equal source values can collapse many subset vectors to one row. Group
    # them before estimating the support; the product of (multiplicity + 1)
    # bounds the number of distinct sums while retaining the actual
    # exponential work bound below.
    multiplicities: dict[Fraction, int] = {}
    for value in (item.as_fraction() for item in nonzero):
        multiplicities[value] = multiplicities.get(value, 0) + 1
    support_upper_bound = 1
    for multiplicity in multiplicities.values():
        support_upper_bound *= multiplicity + 1
    if nonzero:
        lattice_step = 0
        for value in scaled:
            lattice_step = gcd(lattice_step, abs(value))
        support_span = (positive_span - negative_span) // max(lattice_step, 1)
        support_upper_bound = min(support_upper_bound, support_span + 1)
        if common_denominator_overflow:
            if len(nonzero) > 2:
                _reject(
                    "rational_growth_bound",
                    "subset-sum intermediates exceed the canonical rational digit bound",
                )
            exact_sums = {Fraction(0), *fractions, sum(fractions, Fraction(0))}
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
    rational_bytes = strict_json_object_size(
        (("num", growth_digits), ("den", growth_digits))
    )
    row_bytes = strict_json_object_size(
        (("sum_value", rational_bytes), ("multiplicity", len(str(1 << n))))
    )
    predicted_rows = support_upper_bound
    predicted_bytes = source_bytes + 128 + predicted_rows * (row_bytes + 1)
    if predicted_bytes > MAX_RESULT_BYTES:
        _reject(
            "result_size_bound",
            f"the complete subset-sum profile exceeds the {MAX_RESULT_BYTES}-byte output bound",
        )


def compute_rational_subset_sum_profile(
    values: tuple[CanonicalRational, ...],
) -> RationalSubsetSumResult:
    """Return the complete profile of all subset sums of the given rationals.

    For every subset I of the source indices, compute the sum of the
    corresponding rational values. Group by canonical rational value
    and count multiplicities.
    """
    _admit_values(values)
    fractions = [v.as_fraction() for v in values]
    n = len(fractions)
    sum_to_count: dict[Fraction, int] = {}

    for r in range(n + 1):
        for indices in combinations(range(n), r):
            s = sum((fractions[i] for i in indices), Fraction(0))
            sum_to_count[s] = sum_to_count.get(s, 0) + 1

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
