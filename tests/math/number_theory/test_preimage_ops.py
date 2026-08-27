"""Tests for divisor-sum-product fibers and p-adic interval profiles."""

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._preimage_models import (
    DivisorSumProductPreimageRequest,
    PAdicIntervalProfileRequest,
)
from jacobian.math.number_theory._preimage_operations import (
    compute_divisor_sum_product_preimage,
    compute_p_adic_interval_profile,
)


def test_divisor_sum_product_preimage_collision_is_complete() -> None:
    result = compute_divisor_sum_product_preimage(
        DivisorSumProductPreimageRequest(target="336")
    )
    assert result.preimages == ("12", "14")
    assert result.count == 2


def test_divisor_sum_product_preimage_empty_fiber() -> None:
    result = compute_divisor_sum_product_preimage(
        DivisorSumProductPreimageRequest(target="2")
    )
    assert result.preimages == ()


def test_divisor_sum_product_preimage_boundary_is_admitted() -> None:
    request = DivisorSumProductPreimageRequest(target="10000000")
    result = compute_divisor_sum_product_preimage(request)
    assert all(int(n) * int(n) <= 10000000 for n in result.preimages)


def test_divisor_sum_product_preimage_rejects_oversized_target() -> None:
    with pytest.raises(ValidationError):
        DivisorSumProductPreimageRequest(target="10000001")


def test_p_adic_profile_uses_interval_histogram() -> None:
    result = compute_p_adic_interval_profile(
        PAdicIntervalProfileRequest(start="0", length="10", prime="2")
    )
    assert [(row.valuation, row.count) for row in result.rows] == [
        (0, "5"),
        (1, "3"),
        (2, "1"),
        (3, "1"),
    ]
    assert result.total_valuation == "8"
    assert result.maximum_valuation == 3
    assert sum(int(row.count) for row in result.rows) == 10


def test_p_adic_profile_interval_boundaries_and_large_prime() -> None:
    result = compute_p_adic_interval_profile(
        PAdicIntervalProfileRequest(start="7", length="1", prime="2")
    )
    assert [(row.valuation, row.count) for row in result.rows] == [(3, "1")]
    result = compute_p_adic_interval_profile(
        PAdicIntervalProfileRequest(start="0", length="10", prime="101")
    )
    assert [(row.valuation, row.count) for row in result.rows] == [(0, "10")]
    result = compute_p_adic_interval_profile(
        PAdicIntervalProfileRequest(start="0", length="10", prime="1000003")
    )
    assert [(row.valuation, row.count) for row in result.rows] == [(0, "10")]


def test_p_adic_profile_rejects_nonprime_and_empty_interval() -> None:
    nonprime = PAdicIntervalProfileRequest(start="0", length="10", prime="4")
    with pytest.raises(OperationDomainValidationError) as nonprime_error:
        compute_p_adic_interval_profile(nonprime)
    assert nonprime_error.value.errors()[0]["type"] == (
        "number_theory.p_adic_interval_prime_must_be_prime"
    )

    empty = PAdicIntervalProfileRequest(start="0", length="0", prime="2")
    with pytest.raises(OperationDomainValidationError) as empty_error:
        compute_p_adic_interval_profile(empty)
    assert empty_error.value.errors()[0]["type"] == (
        "number_theory.p_adic_interval_length_must_be_positive"
    )


def test_p_adic_profile_rejects_endpoint_overflow_after_strict_parsing() -> None:
    request = PAdicIntervalProfileRequest(start="9" * 252, length="1", prime="2")
    with pytest.raises(OperationDomainValidationError) as error:
        compute_p_adic_interval_profile(request)
    assert error.value.errors()[0]["type"] == (
        "number_theory.p_adic_interval_endpoint_digits"
    )


def test_p_adic_profile_matches_direct_small_interval() -> None:
    result = compute_p_adic_interval_profile(
        PAdicIntervalProfileRequest(start="3", length="8", prime="3")
    )
    expected: dict[int, int] = {}
    for n in range(4, 12):
        value = n
        valuation = 0
        while value % 3 == 0:
            valuation += 1
            value //= 3
        expected[valuation] = expected.get(valuation, 0) + 1
    assert {row.valuation: int(row.count) for row in result.rows} == expected
