"""Tests for gcd-quotient and product-divisibility profiles."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.math.number_theory._divisibility_profile_models import (
    GcdQuotientProfileRequest,
    ProductDivisibilityProfileRequest,
)
from jacobian.math.number_theory._divisibility_profile_operations import (
    compute_gcd_quotient_profile,
    compute_product_divisibility_profile,
)


def test_gcd_quotient_basic() -> None:
    result = compute_gcd_quotient_profile(
        GcdQuotientProfileRequest(elements=("6", "10", "15"))
    )
    vals = [int(e) for e in result.elements]
    assert vals == [6, 10, 15]
    assert result.quotients[0][0].as_fraction() == Fraction(1)  # gcd(6,6) / 6
    assert result.quotients[0][1].as_fraction() == Fraction(1, 5)  # gcd(6,10) / 10
    assert result.quotients[1][2].as_fraction() == Fraction(1, 3)  # gcd(10,15) / 15
    assert result.quotients[0][2].as_fraction() == Fraction(1, 5)  # gcd(6,15) / 15


def test_gcd_quotient_symmetric() -> None:
    result = compute_gcd_quotient_profile(
        GcdQuotientProfileRequest(elements=("12", "18", "24"))
    )
    for i in range(3):
        for j in range(3):
            assert result.quotients[i][j] == result.quotients[j][i]


def test_product_divisibility_basic() -> None:
    result = compute_product_divisibility_profile(
        ProductDivisibilityProfileRequest(elements=("2", "3"))
    )
    assert result.divisibility_matrix == ((False, True), (True, False))


def test_profile_requests_reject_nonpositive_elements() -> None:
    with pytest.raises(ValidationError, match="must be positive"):
        GcdQuotientProfileRequest(elements=("0",))
    with pytest.raises(ValidationError, match="must be positive"):
        ProductDivisibilityProfileRequest(elements=("-2",))
