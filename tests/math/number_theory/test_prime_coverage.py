"""Tests for prime-coverage and binomial-valuation profiles."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._additional_ops import (
    compute_binomial_valuation_profile,
    compute_prime_coverage_profile,
)
from jacobian.math.number_theory._binomial_valuation_models import (
    BinomialValuationProfileRequest,
)
from jacobian.math.number_theory._prime_coverage_models import (
    PrimeCoverageProfileRequest,
)
from jacobian.math.number_theory.operations import (
    binomial_valuation_profile,
    prime_coverage_profile,
)


def test_prime_coverage_small() -> None:
    result = compute_prime_coverage_profile(
        PrimeCoverageProfileRequest(lower_bound=1, upper_bound=10)
    )
    counts = [r.distinct_prime_count for r in result.rows]
    # omega(1)=0, omega(2)=1, omega(3)=1, omega(4)=1, omega(5)=1, omega(6)=2, omega(7)=1, omega(8)=1, omega(9)=1, omega(10)=2
    assert counts == [0, 1, 1, 1, 1, 2, 1, 1, 1, 2]


def test_native_profiles_accept_canonical_values() -> None:
    assert len(prime_coverage_profile(1, 3).rows) == 3
    assert [row.valuation for row in binomial_valuation_profile(4, 2).rows] == [
        0,
        2,
        1,
        2,
        0,
    ]


def test_prime_coverage_primes() -> None:
    result = compute_prime_coverage_profile(
        PrimeCoverageProfileRequest(lower_bound=2, upper_bound=50)
    )
    for row in result.rows:
        from sympy import isprime

        if isprime(row.n):
            assert row.distinct_prime_count == 1


def test_prime_coverage_admits_narrow_high_interval() -> None:
    request = PrimeCoverageProfileRequest(
        lower_bound=10_000_001,
        upper_bound=10_000_001,
    )
    result = compute_prime_coverage_profile(request)

    assert [(row.n, row.distinct_prime_count) for row in result.rows] == [
        (10_000_001, 2)
    ]


def test_prime_coverage_rejects_result_over_canonical_output_budget() -> None:
    request = PrimeCoverageProfileRequest(lower_bound=1, upper_bound=1_000_000)

    with pytest.raises(OperationDomainValidationError, match="canonical output budget"):
        compute_prime_coverage_profile(request)


def test_prime_coverage_rejects_square_root_work_budget() -> None:
    request = PrimeCoverageProfileRequest(lower_bound=10**13, upper_bound=10**13)

    with pytest.raises(
        OperationDomainValidationError, match="segmented prime-coverage work budget"
    ):
        compute_prime_coverage_profile(request)


def test_binomial_valuation_rejects_composite_base() -> None:
    request = BinomialValuationProfileRequest(n=4, prime=4)

    with pytest.raises(OperationDomainValidationError, match="prime must be"):
        compute_binomial_valuation_profile(request)


def test_binomial_valuation_admits_output_bound_n_and_large_prime() -> None:
    result = compute_binomial_valuation_profile(
        BinomialValuationProfileRequest(n=1001, prime=2)
    )
    assert len(result.rows) == 1002

    large_base_result = compute_binomial_valuation_profile(
        BinomialValuationProfileRequest(n=10, prime=10_007)
    )
    assert [row.valuation for row in large_base_result.rows] == [0] * 11


def test_binomial_valuation_rejects_digitwise_work_budget() -> None:
    request = BinomialValuationProfileRequest(n=120_000, prime=2)

    with pytest.raises(OperationDomainValidationError, match="digitwise work budget"):
        compute_binomial_valuation_profile(request)


def test_binomial_valuation_basic() -> None:
    # v_2(C(4,2)) = v_2(6) = 1
    result = compute_binomial_valuation_profile(
        BinomialValuationProfileRequest(n=4, prime=2)
    )
    assert result.rows[2].valuation == 1


def test_binomial_valuation_kummer() -> None:
    # v_2(C(10,3)) = v_2(120) = 3
    result = compute_binomial_valuation_profile(
        BinomialValuationProfileRequest(n=10, prime=2)
    )
    assert result.rows[3].valuation == 3
    # v_2(C(10,5)) = v_2(252) = 2
    assert result.rows[5].valuation == 2


def test_binomial_valuation_matches_sympy() -> None:
    from sympy import binomial

    result = compute_binomial_valuation_profile(
        BinomialValuationProfileRequest(n=8, prime=3)
    )
    for row in result.rows:
        val = int(binomial(8, row.k))
        # Count factor of 3 in val
        expected = 0
        temp = val
        while temp % 3 == 0:
            expected += 1
            temp //= 3
        assert row.valuation == expected
