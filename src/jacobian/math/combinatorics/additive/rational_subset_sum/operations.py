"""Rational subset-sum profile kernel."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import NoReturn

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    canonical_rational_component_digits,
)
from jacobian.canonical import CanonicalLimits, encode_strict_json
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

    max_digits = max(canonical_rational_component_digits(value) for value in values)
    # Adding n fractions can multiply denominators and numerators by every
    # other input denominator.  This is a conservative bound for every
    # intermediate subset sum, including the complete sum.
    growth_digits = n * max_digits + len(str(n)) + 1
    if growth_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        _reject(
            "rational_growth_bound",
            "subset-sum intermediates exceed the canonical rational digit bound",
        )

    source_bytes = len(
        encode_strict_json({"values": [v.model_dump(mode="json") for v in values]})
    )
    row_bytes = 2 * growth_digits + 96
    predicted_rows = 1 << n
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
