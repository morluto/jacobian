from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.weighted_monotone_profiles.operations import (
    compute_weighted_monotone_profiles,
)


def _cr(num: int, den: int = 1) -> CanonicalRational:
    return CanonicalRational.from_fraction(Fraction(num, den))


def test_single_element() -> None:
    result = compute_weighted_monotone_profiles((5,), (_cr(3),))
    assert result.increasing_profile[0].as_fraction() == Fraction(3)
    assert result.decreasing_profile[0].as_fraction() == Fraction(3)


def test_increasing_alphabet() -> None:
    result = compute_weighted_monotone_profiles(
        (1, 2, 3),
        (_cr(1), _cr(2), _cr(3)),
    )
    inc = [v.as_fraction() for v in result.increasing_profile]
    # All can chain: 1, 1+2=3, 3+3=6
    assert inc == [Fraction(1), Fraction(3), Fraction(6)]


def test_decreasing_alphabet() -> None:
    result = compute_weighted_monotone_profiles(
        (3, 2, 1),
        (_cr(1), _cr(2), _cr(3)),
    )
    dec = [v.as_fraction() for v in result.decreasing_profile]
    # All can chain: 1, 1+2=3, 3+3=6
    assert dec == [Fraction(1), Fraction(3), Fraction(6)]


def test_mixed() -> None:
    result = compute_weighted_monotone_profiles(
        (3, 1, 2, 1, 3),
        (_cr(1), _cr(2), _cr(3), _cr(1), _cr(4)),
    )
    inc = [v.as_fraction() for v in result.increasing_profile]
    # Increasing: can chain when a[j] <= a[i]
    # i=0: S=1 (w=1, no prior)
    # i=1: a[1]=1, no j<1 with a[j]<=1 (a[0]=3 > 1), S=2
    # i=2: a[2]=2, j=0 (3<=2? no), j=1 (1<=2? yes, S=2), S=3+2=5
    # i=3: a[3]=1, j=0 (3<=1? no), j=1 (1<=1? yes, S=2), j=2 (2<=1? no), S=1+2=3
    # i=4: a[4]=3, j=0 (3<=3 yes S=1), j=1 (1<=3 yes S=2), j=2 (2<=3 yes S=5), j=3 (1<=3 yes S=3), best=5, S=4+5=9
    assert inc[0] == Fraction(1)
    assert inc[1] == Fraction(2)
    assert inc[2] == Fraction(5)
    assert inc[3] == Fraction(3)
    assert inc[4] == Fraction(9)


def test_result_preserves_source() -> None:
    result = compute_weighted_monotone_profiles(
        (1, 2),
        (_cr(1), _cr(1)),
    )
    assert result.alphabet == (1, 2)
    assert len(result.weights) == 2


def test_weight_axis_must_match_the_alphabet() -> None:
    with pytest.raises(OperationDomainValidationError, match="one-for-one"):
        compute_weighted_monotone_profiles((1,), ())
