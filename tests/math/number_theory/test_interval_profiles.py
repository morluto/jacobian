"""Contract tests for exact integer-interval arithmetic-function profiles."""

from __future__ import annotations

import pytest
from sympy import factorint, isprime

from jacobian.math.number_theory._interval_profile_models import (
    IntervalProfileRequest,
    IntervalProfileRowsRequest,
    SquarefreeProfileRequest,
)
from jacobian.math.number_theory._interval_profile_operations import (
    compute_divisor_count_profile,
    compute_divisor_sum_profile,
    compute_euler_totient_profile,
    compute_greatest_prime_factor_profile,
    compute_least_prime_factor_profile,
    compute_prime_gap_profile,
    compute_squarefree_profile,
)


class TestSquarefreeProfile:
    def test_request_rejects_reversed_interval(self) -> None:
        with pytest.raises(ValueError, match="upper_bound must be >= lower_bound"):
            IntervalProfileRequest(lower_bound=5, upper_bound=3)

    def test_request_rejects_result_larger_than_canonical_budget(self) -> None:
        with pytest.raises(ValueError, match="canonical output budget"):
            IntervalProfileRowsRequest(lower_bound=1, upper_bound=300_000)

    def test_small_interval(self) -> None:
        result = compute_squarefree_profile(
            IntervalProfileRequest(lower_bound=1, upper_bound=12)
        )
        assert list(result.squarefree_values) == [1, 2, 3, 5, 6, 7, 10, 11]
        assert list(result.nonsquarefree_values) == [4, 8, 9, 12]
        assert result.squarefree_count == 8
        assert result.nonsquarefree_count == 4

    def test_partition_exhaustive(self) -> None:
        lo, hi = 10, 100
        result = compute_squarefree_profile(
            IntervalProfileRequest(lower_bound=lo, upper_bound=hi)
        )
        assert sorted(result.squarefree_values + result.nonsquarefree_values) == list(
            range(lo, hi + 1)
        )
        assert set(result.squarefree_values) & set(result.nonsquarefree_values) == set()

    def test_matches_factorization(self) -> None:
        lo, hi = 1, 200
        result = compute_squarefree_profile(
            IntervalProfileRequest(lower_bound=lo, upper_bound=hi)
        )
        expected_sf = [
            n for n in range(lo, hi + 1) if all(e == 1 for _, e in factorint(n).items())
        ]
        assert list(result.squarefree_values) == expected_sf


class TestDivisorCountProfile:
    def test_small_interval(self) -> None:
        result = compute_divisor_count_profile(
            IntervalProfileRequest(lower_bound=1, upper_bound=12)
        )
        assert [r.divisor_count for r in result.rows] == [
            1,
            2,
            2,
            3,
            2,
            4,
            2,
            4,
            3,
            4,
            2,
            6,
        ]

    def test_prime_powers(self) -> None:
        result = compute_divisor_count_profile(
            IntervalProfileRequest(lower_bound=1, upper_bound=100)
        )
        rows = {r.n: r.divisor_count for r in result.rows}
        assert rows[4] == 3
        assert rows[8] == 4
        assert rows[9] == 3
        assert rows[16] == 5
        assert rows[64] == 7


class TestGreatestPrimeFactorProfile:
    def test_small_interval(self) -> None:
        result = compute_greatest_prime_factor_profile(
            IntervalProfileRequest(lower_bound=1, upper_bound=10)
        )
        assert [r.greatest_prime_factor for r in result.rows] == [
            1,
            2,
            3,
            2,
            5,
            3,
            7,
            2,
            3,
            5,
        ]

    def test_matches_factorization(self) -> None:
        result = compute_greatest_prime_factor_profile(
            IntervalProfileRequest(lower_bound=1, upper_bound=200)
        )
        for row in result.rows:
            if row.n == 1:
                assert row.greatest_prime_factor == 1
            else:
                assert row.greatest_prime_factor == max(factorint(row.n))


class TestPrimeGapProfile:
    def test_small_interval(self) -> None:
        result = compute_prime_gap_profile(
            IntervalProfileRequest(lower_bound=3, upper_bound=5)
        )
        assert len(result.rows) == 2
        assert result.rows[0].lower_prime == 3
        assert result.rows[0].upper_prime == 5
        assert result.rows[0].gap == 2
        assert result.rows[1].lower_prime == 5
        assert result.rows[1].upper_prime == 7
        assert result.rows[1].gap == 2

    def test_successor_beyond_upper(self) -> None:
        result = compute_prime_gap_profile(
            IntervalProfileRequest(lower_bound=5, upper_bound=5)
        )
        assert len(result.rows) == 1
        assert result.rows[0].lower_prime == 5
        assert result.rows[0].upper_prime == 7

    def test_consecutive_primes(self) -> None:
        result = compute_prime_gap_profile(
            IntervalProfileRequest(lower_bound=2, upper_bound=100)
        )
        for row in result.rows:
            assert isprime(row.lower_prime)
            assert isprime(row.upper_prime)
            for p in range(row.lower_prime + 1, row.upper_prime):
                assert not isprime(p)


class TestLeastPrimeFactorProfile:
    def test_small_interval(self) -> None:
        result = compute_least_prime_factor_profile(
            IntervalProfileRequest(lower_bound=1, upper_bound=10)
        )
        assert [r.least_prime_factor for r in result.rows] == [
            1,
            2,
            3,
            2,
            5,
            2,
            7,
            2,
            3,
            2,
        ]

    def test_primes_return_themselves(self) -> None:
        result = compute_least_prime_factor_profile(
            IntervalProfileRequest(lower_bound=2, upper_bound=50)
        )
        for row in result.rows:
            if isprime(row.n):
                assert row.least_prime_factor == row.n


class TestEulerTotientProfile:
    def test_small_interval(self) -> None:
        result = compute_euler_totient_profile(
            IntervalProfileRequest(lower_bound=1, upper_bound=10)
        )
        assert [r.euler_totient for r in result.rows] == [1, 1, 2, 2, 4, 2, 6, 4, 6, 4]

    def test_matches_sympy(self) -> None:
        from sympy import totient

        result = compute_euler_totient_profile(
            IntervalProfileRequest(lower_bound=1, upper_bound=100)
        )
        for row in result.rows:
            assert row.euler_totient == int(totient(row.n))


class TestDivisorSumProfile:
    def test_small_interval(self) -> None:
        result = compute_divisor_sum_profile(
            IntervalProfileRequest(lower_bound=1, upper_bound=6)
        )
        assert [r.divisor_sum for r in result.rows] == [1, 3, 4, 7, 6, 12]

    def test_matches_sympy(self) -> None:
        from sympy import divisor_sigma

        result = compute_divisor_sum_profile(
            IntervalProfileRequest(lower_bound=1, upper_bound=100)
        )
        for row in result.rows:
            assert row.divisor_sum == int(divisor_sigma(row.n))


def test_narrow_high_interval_uses_exact_segmented_profiles() -> None:
    request = IntervalProfileRequest(
        lower_bound=10_000_000,
        upper_bound=10_000_000,
    )

    assert compute_divisor_count_profile(request).rows[0].divisor_count == 64
    assert (
        compute_greatest_prime_factor_profile(request).rows[0].greatest_prime_factor
        == 5
    )
    assert compute_least_prime_factor_profile(request).rows[0].least_prime_factor == 2
    assert compute_euler_totient_profile(request).rows[0].euler_totient == 4_000_000
    assert compute_divisor_sum_profile(request).rows[0].divisor_sum == 24_902_280
    assert compute_prime_gap_profile(request).rows == ()


def test_compact_squarefree_request_uses_its_result_shape() -> None:
    request = SquarefreeProfileRequest(lower_bound=1, upper_bound=1_000_000)
    assert request.width() == 1_000_000
