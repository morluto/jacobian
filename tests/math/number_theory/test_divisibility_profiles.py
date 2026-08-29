"""Tests for gcd-quotient and product-divisibility profiles."""

from fractions import Fraction

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._divisibility_profile_models import (
    GcdQuotientProfileRequest,
    ProductDivisibilityProfileRequest,
)
from jacobian.math.number_theory._divisibility_profiles import (
    compute_gcd_quotient_profile,
    compute_product_divisibility_profile,
)
from jacobian.math.number_theory.operations import (
    gcd_quotient_profile,
    product_divisibility_profile,
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


def test_native_profiles_accept_canonical_values() -> None:
    assert gcd_quotient_profile((6, 10)).elements == ("6", "10")
    assert product_divisibility_profile((2, 3)).divisibility_matrix == (
        (False, True),
        (True, False),
    )


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
    with pytest.raises(OperationDomainValidationError, match="must be positive"):
        compute_gcd_quotient_profile(GcdQuotientProfileRequest(elements=("0",)))
    with pytest.raises(OperationDomainValidationError, match="must be positive"):
        compute_product_divisibility_profile(
            ProductDivisibilityProfileRequest(elements=("-2",))
        )


def test_profile_declarations_publish_positive_integer_domain() -> None:
    from jacobian.math.number_theory._divisibility_profiles import (
        DIVISIBILITY_PROFILE_OPERATIONS,
    )

    schemas = [
        GcdQuotientProfileRequest.model_json_schema(),
        ProductDivisibilityProfileRequest.model_json_schema(),
    ]
    for schema in schemas:
        description = schema["properties"]["elements"]["description"]
        assert "positive integers" in description
        assert "Zero and negative integers are invalid" in description

    for operation in DIVISIBILITY_PROFILE_OPERATIONS:
        assert "positive integers" in operation.description
        assert "Zero and negative integers are outside the request domain" in (
            operation.description
        )
