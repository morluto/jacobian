from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive.rational_subset_sum.operations import (
    compute_rational_subset_sum_profile,
)


def _cr(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(num, den))


def test_empty() -> None:
    result = compute_rational_subset_sum_profile(())
    assert result.support_cardinality == 1  # Only the empty subset sum = 0
    assert result.entries[0].sum.as_fraction() == Fraction(0)


def test_single_value() -> None:
    result = compute_rational_subset_sum_profile((_cr(1, 2),))
    entries = {e.sum.as_fraction(): e.multiplicity for e in result.entries}
    assert entries == {Fraction(0): 1, Fraction(1, 2): 1}


def test_two_values() -> None:
    result = compute_rational_subset_sum_profile((_cr(1, 2), _cr(1, 3)))
    entries = {e.sum.as_fraction(): e.multiplicity for e in result.entries}
    # Subsets: {}=0, {0}=1/2, {1}=1/3, {0,1}=5/6
    assert entries == {
        Fraction(0): 1,
        Fraction(1, 3): 1,
        Fraction(1, 2): 1,
        Fraction(5, 6): 1,
    }


def test_duplicate_sums() -> None:
    """{1, -1} has sum 0 for {} and {0,1}."""
    result = compute_rational_subset_sum_profile((_cr(1), _cr(-1)))
    entries = {e.sum.as_fraction(): e.multiplicity for e in result.entries}
    assert entries[Fraction(0)] == 2  # Empty set and full set


def test_result_preserves_source() -> None:
    values = (_cr(1), _cr(2))
    result = compute_rational_subset_sum_profile(values)
    assert result.values == values


def test_subset_enumeration_work_is_bounded() -> None:
    values = tuple(_cr(0) for _ in range(17))

    with pytest.raises(OperationDomainValidationError, match="subset work bound"):
        compute_rational_subset_sum_profile(values)


def test_derived_subset_sum_must_fit_the_rational_carrier() -> None:
    denominator = 10**32_767
    values = (_cr(1, denominator), _cr(1, denominator - 1))

    with pytest.raises(OperationDomainValidationError, match="derived subset sum"):
        compute_rational_subset_sum_profile(values)
