"""Tests for gcd-quotient and product-divisibility profiles."""

from jacobian.math.number_theory._divisibility_profile_models import (
    GcdQuotientProfileRequest,
    ProductDivisibilityProfileRequest,
)
from jacobian.math.number_theory._divisibility_profile_operations import (
    compute_gcd_quotient_profile,
    compute_product_divisibility_profile,
)
import math


def test_gcd_quotient_basic() -> None:
    result = compute_gcd_quotient_profile(GcdQuotientProfileRequest(elements=["6", "10", "15"]))
    vals = [int(e) for e in result.elements]
    assert vals == [6, 10, 15]
    assert result.quotients[0][0] == 6  # gcd(6,6) = 6
    assert result.quotients[0][1] == 2  # gcd(6,10) = 2
    assert result.quotients[1][2] == 5  # gcd(10,15) = 5
    assert result.quotients[0][2] == 3  # gcd(6,15) = 3


def test_gcd_quotient_symmetric() -> None:
    result = compute_gcd_quotient_profile(GcdQuotientProfileRequest(elements=["12", "18", "24"]))
    for i in range(3):
        for j in range(3):
            assert result.quotients[i][j] == result.quotients[j][i]


def test_product_divisibility_basic() -> None:
    result = compute_product_divisibility_profile(ProductDivisibilityProfileRequest(elements=["2", "6", "12"]))
    # 2 divides 6? yes. 2 divides 12? yes. 6 divides 12? yes.
    assert result.divisibility_matrix[0][1] == True  # 2|6
    assert result.divisibility_matrix[0][2] == True  # 2|12
    assert result.divisibility_matrix[1][2] == True  # 6|12
    assert result.divisibility_matrix[1][0] == False  # 6 does not divide 2
