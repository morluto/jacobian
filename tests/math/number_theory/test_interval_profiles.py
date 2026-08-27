"""Contract tests for exact integer-interval arithmetic-function profiles."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from sympy import factorint, isprime

from jacobian.math.number_theory._interval_profile_models import (
    MAX_INTERVAL_WIDTH,
    MAX_PROFILE_RESULT_BYTES,
    MAX_SIEVE_WORK,
    DivisorCountProfileRequest,
    DivisorSumProfileRequest,
    EulerTotientProfileRequest,
    GreatestPrimeFactorProfileRequest,
    IntervalProfileRequest,
    IntervalProfileRowsRequest,
    LeastPrimeFactorProfileRequest,
    PrimeGapProfileRequest,
    SquarefreeProfileRequest,
    _estimate_prime_gap_work,
    _estimate_successor_prime_search_work,
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

# ---------------------------------------------------------------------------
# Squarefree profile
# ---------------------------------------------------------------------------


def _is_squarefree(n: int) -> bool:
    if n <= 0:
        return False
    return all(exp == 1 for _, exp in factorint(n).items())


class TestSquarefreeProfile:
    def test_small_interval(self) -> None:
        result = compute_squarefree_profile(
            SquarefreeProfileRequest(lower_bound=1, upper_bound=12)
        )
        assert list(result.squarefree_values) == [1, 2, 3, 5, 6, 7, 10, 11]
        assert list(result.nonsquarefree_values) == [4, 8, 9, 12]
        assert result.squarefree_count == 8
        assert result.nonsquarefree_count == 4

    def test_single_element(self) -> None:
        result = compute_squarefree_profile(
            SquarefreeProfileRequest(lower_bound=1, upper_bound=1)
        )
        assert list(result.squarefree_values) == [1]
        assert list(result.nonsquarefree_values) == []
        assert result.squarefree_count == 1
        assert result.nonsquarefree_count == 0

    def test_no_squarefree(self) -> None:
        """Interval [4, 4] contains only 4, which is non-squarefree."""
        result = compute_squarefree_profile(
            SquarefreeProfileRequest(lower_bound=4, upper_bound=4)
        )
        assert list(result.squarefree_values) == []
        assert list(result.nonsquarefree_values) == [4]

    def test_profile_members_are_immutable(self) -> None:
        result = compute_squarefree_profile(
            SquarefreeProfileRequest(lower_bound=1, upper_bound=4)
        )
        with pytest.raises(AttributeError):
            result.squarefree_values.append(5)  # type: ignore[attr-defined]

    def test_partition_is_exhaustive_and_disjoint(self) -> None:
        lo, hi = 10, 100
        result = compute_squarefree_profile(
            SquarefreeProfileRequest(lower_bound=lo, upper_bound=hi)
        )
        sf = result.squarefree_values
        nsf = result.nonsquarefree_values
        # Union is the full interval
        assert sorted(sf + nsf) == list(range(lo, hi + 1))
        # Disjoint
        assert set(sf) & set(nsf) == set()
        # Exhaustive
        assert len(sf) + len(nsf) == hi - lo + 1

    def test_matches_direct_factorization(self) -> None:
        lo, hi = 1, 200
        result = compute_squarefree_profile(
            SquarefreeProfileRequest(lower_bound=lo, upper_bound=hi)
        )
        expected_sf = [n for n in range(lo, hi + 1) if _is_squarefree(n)]
        expected_nsf = [n for n in range(lo, hi + 1) if not _is_squarefree(n)]
        assert list(result.squarefree_values) == expected_sf
        assert list(result.nonsquarefree_values) == expected_nsf

    def test_request_rejects_reversed_interval(self) -> None:
        with pytest.raises(ValidationError, match="upper_bound must be >= lower_bound"):
            IntervalProfileRequest(lower_bound=2, upper_bound=1)

    def test_request_rejects_overwide_interval(self) -> None:
        with pytest.raises(ValidationError, match="interval width exceeds"):
            IntervalProfileRequest(
                lower_bound=1,
                upper_bound=MAX_INTERVAL_WIDTH + 1,
            )

    def test_request_rejects_result_over_canonical_budget(self) -> None:
        with pytest.raises(ValidationError, match="canonical output budget"):
            DivisorCountProfileRequest(
                lower_bound=1,
                upper_bound=MAX_INTERVAL_WIDTH,
            )

    def test_operation_specific_result_bounds_preserve_sparse_profiles(self) -> None:
        sparse_width = MAX_INTERVAL_WIDTH + 1
        squarefree = SquarefreeProfileRequest(lower_bound=1, upper_bound=sparse_width)
        prime_gap = PrimeGapProfileRequest(lower_bound=1, upper_bound=sparse_width)

        assert squarefree.width() == sparse_width
        assert prime_gap.width() == sparse_width
        assert squarefree.admission.estimated_result_bytes <= MAX_PROFILE_RESULT_BYTES
        assert prime_gap.admission.estimated_result_bytes <= MAX_PROFILE_RESULT_BYTES
        with pytest.raises(ValidationError, match="canonical output budget"):
            DivisorCountProfileRequest(lower_bound=1, upper_bound=MAX_INTERVAL_WIDTH)
        with pytest.raises(ValidationError, match="canonical output budget"):
            GreatestPrimeFactorProfileRequest(
                lower_bound=1, upper_bound=MAX_INTERVAL_WIDTH
            )

    def test_narrow_high_intervals_use_work_and_result_budgets(self) -> None:
        requests = (
            SquarefreeProfileRequest,
            DivisorCountProfileRequest,
            GreatestPrimeFactorProfileRequest,
            PrimeGapProfileRequest,
        )

        for request_type in requests:
            request = request_type(lower_bound=10_000_001, upper_bound=10_000_001)
            assert request.admission.estimated_work <= MAX_SIEVE_WORK
            assert request.admission.estimated_result_bytes <= MAX_PROFILE_RESULT_BYTES

    def test_prime_gap_bounds_rows_by_interval_density(self) -> None:
        request = PrimeGapProfileRequest(
            lower_bound=9_000_001,
            upper_bound=10_000_000,
        )

        assert request.admission.estimated_result_bytes <= MAX_PROFILE_RESULT_BYTES

    def test_prime_gap_work_charges_successor_search(self) -> None:
        request = PrimeGapProfileRequest(lower_bound=1, upper_bound=1_000_001)

        assert request.admission.estimated_work == _estimate_prime_gap_work(
            request.lower_bound, request.upper_bound
        )
        assert _estimate_successor_prime_search_work(request.upper_bound) > 0

    def test_kernels_reject_a_different_operation_admission(self) -> None:
        sparse_request = SquarefreeProfileRequest(
            lower_bound=1, upper_bound=MAX_INTERVAL_WIDTH + 1
        )

        with pytest.raises(
            TypeError, match="expected GreatestPrimeFactorProfileRequest"
        ):
            compute_greatest_prime_factor_profile(sparse_request)  # type: ignore[arg-type]

    def test_work_budget_replaces_fixed_upper_bound(self) -> None:
        with pytest.raises(ValidationError, match="segmented-sieve work budget"):
            SquarefreeProfileRequest(lower_bound=10**13, upper_bound=10**13)

    def test_prime_square_boundary(self) -> None:
        """4 = 2^2 is the first non-squarefree, 9 = 3^2 is another."""
        result = compute_squarefree_profile(
            SquarefreeProfileRequest(lower_bound=1, upper_bound=20)
        )
        assert 4 in result.nonsquarefree_values
        assert 8 in result.nonsquarefree_values  # 2^3
        assert 9 in result.nonsquarefree_values  # 3^2
        assert 12 in result.nonsquarefree_values  # 2^2 * 3
        assert 16 in result.nonsquarefree_values  # 2^4
        assert 18 in result.nonsquarefree_values  # 2 * 3^2
        assert 20 in result.nonsquarefree_values  # 2^2 * 5

    def test_concatenation_consistency(self) -> None:
        r1 = compute_squarefree_profile(
            SquarefreeProfileRequest(lower_bound=1, upper_bound=10)
        )
        r2 = compute_squarefree_profile(
            SquarefreeProfileRequest(lower_bound=11, upper_bound=20)
        )
        r_union = compute_squarefree_profile(
            SquarefreeProfileRequest(lower_bound=1, upper_bound=20)
        )
        assert r1.squarefree_values + r2.squarefree_values == r_union.squarefree_values
        assert (
            r1.nonsquarefree_values + r2.nonsquarefree_values
            == r_union.nonsquarefree_values
        )


# ---------------------------------------------------------------------------
# Divisor-count profile
# ---------------------------------------------------------------------------


class TestDivisorCountProfile:
    def test_small_interval(self) -> None:
        result = compute_divisor_count_profile(
            DivisorCountProfileRequest(lower_bound=1, upper_bound=12)
        )
        expected = [1, 2, 2, 3, 2, 4, 2, 4, 3, 4, 2, 6]
        assert [r.divisor_count for r in result.rows] == expected

    def test_single_element(self) -> None:
        result = compute_divisor_count_profile(
            DivisorCountProfileRequest(lower_bound=1, upper_bound=1)
        )
        assert result.rows[0].n == 1
        assert result.rows[0].divisor_count == 1

    def test_primes_have_count_two(self) -> None:
        result = compute_divisor_count_profile(
            DivisorCountProfileRequest(lower_bound=2, upper_bound=50)
        )
        for row in result.rows:
            if isprime(row.n):
                assert row.divisor_count == 2

    def test_prime_powers(self) -> None:
        """tau(p^e) = e + 1."""
        result = compute_divisor_count_profile(
            DivisorCountProfileRequest(lower_bound=1, upper_bound=100)
        )
        rows = {r.n: r.divisor_count for r in result.rows}
        assert rows[4] == 3  # 2^2
        assert rows[8] == 4  # 2^3
        assert rows[9] == 3  # 3^2
        assert rows[16] == 5  # 2^4
        assert rows[27] == 4  # 3^3
        assert rows[32] == 6  # 2^5
        assert rows[64] == 7  # 2^6

    def test_matches_direct_divisor_enumeration(self) -> None:
        lo, hi = 1, 200
        result = compute_divisor_count_profile(
            DivisorCountProfileRequest(lower_bound=lo, upper_bound=hi)
        )
        for row in result.rows:
            n = row.n
            expected_tau = len([d for d in range(1, n + 1) if n % d == 0])
            assert row.divisor_count == expected_tau

    def test_coverage(self) -> None:
        lo, hi = 5, 50
        result = compute_divisor_count_profile(
            DivisorCountProfileRequest(lower_bound=lo, upper_bound=hi)
        )
        assert [r.n for r in result.rows] == list(range(lo, hi + 1))


# ---------------------------------------------------------------------------
# Greatest-prime-factor profile
# ---------------------------------------------------------------------------


class TestGreatestPrimeFactorProfile:
    def test_small_interval(self) -> None:
        result = compute_greatest_prime_factor_profile(
            GreatestPrimeFactorProfileRequest(lower_bound=1, upper_bound=10)
        )
        expected = [1, 2, 3, 2, 5, 3, 7, 2, 3, 5]
        assert [r.greatest_prime_factor for r in result.rows] == expected

    def test_single_element(self) -> None:
        result = compute_greatest_prime_factor_profile(
            GreatestPrimeFactorProfileRequest(lower_bound=1, upper_bound=1)
        )
        assert result.rows[0].n == 1
        assert result.rows[0].greatest_prime_factor == 1

    def test_prime_interval(self) -> None:
        """A prime interval returns each integer as its own greatest prime factor."""
        result = compute_greatest_prime_factor_profile(
            GreatestPrimeFactorProfileRequest(lower_bound=2, upper_bound=2)
        )
        assert result.rows[0].greatest_prime_factor == 2

    def test_matches_direct_factorization(self) -> None:
        lo, hi = 1, 200
        result = compute_greatest_prime_factor_profile(
            GreatestPrimeFactorProfileRequest(lower_bound=lo, upper_bound=hi)
        )
        for row in result.rows:
            n = row.n
            if n == 1:
                assert row.greatest_prime_factor == 1
                continue
            factors = factorint(n)
            assert row.greatest_prime_factor == max(factors)

    def test_prime_powers(self) -> None:
        """For p^e, the greatest prime factor is p."""
        result = compute_greatest_prime_factor_profile(
            GreatestPrimeFactorProfileRequest(lower_bound=1, upper_bound=100)
        )
        rows = {r.n: r.greatest_prime_factor for r in result.rows}
        assert rows[4] == 2  # 2^2
        assert rows[8] == 2  # 2^3
        assert rows[9] == 3  # 3^2
        assert rows[27] == 3  # 3^3
        assert rows[32] == 2  # 2^5

    def test_coverage(self) -> None:
        lo, hi = 5, 50
        result = compute_greatest_prime_factor_profile(
            GreatestPrimeFactorProfileRequest(lower_bound=lo, upper_bound=hi)
        )
        assert [r.n for r in result.rows] == list(range(lo, hi + 1))


# ---------------------------------------------------------------------------
# Prime-gap profile
# ---------------------------------------------------------------------------


class TestPrimeGapProfile:
    def test_small_interval(self) -> None:
        result = compute_prime_gap_profile(
            PrimeGapProfileRequest(lower_bound=3, upper_bound=5)
        )
        assert len(result.rows) == 2
        assert result.rows[0].lower_prime == 3
        assert result.rows[0].upper_prime == 5
        assert result.rows[0].gap == 2
        assert result.rows[1].lower_prime == 5
        assert result.rows[1].upper_prime == 7
        assert result.rows[1].gap == 2

    def test_single_prime(self) -> None:
        """[2, 2] returns (2, 3, 1)."""
        result = compute_prime_gap_profile(
            PrimeGapProfileRequest(lower_bound=2, upper_bound=2)
        )
        assert len(result.rows) == 1
        assert result.rows[0].lower_prime == 2
        assert result.rows[0].upper_prime == 3
        assert result.rows[0].gap == 1

    def test_no_primes(self) -> None:
        """[4, 4] has no primes, so the gap list is empty."""
        result = compute_prime_gap_profile(
            PrimeGapProfileRequest(lower_bound=4, upper_bound=4)
        )
        assert len(result.rows) == 0

    def test_successor_beyond_upper(self) -> None:
        """The final gap whose lower endpoint is in [L, U] must include
        the successor prime beyond U."""
        result = compute_prime_gap_profile(
            PrimeGapProfileRequest(lower_bound=5, upper_bound=5)
        )
        assert len(result.rows) == 1
        assert result.rows[0].lower_prime == 5
        assert result.rows[0].upper_prime == 7
        assert result.rows[0].gap == 2

    def test_rows_are_consecutive_primes(self) -> None:
        """Every row consists of two primes with no prime between them."""
        result = compute_prime_gap_profile(
            PrimeGapProfileRequest(lower_bound=2, upper_bound=100)
        )
        for row in result.rows:
            assert isprime(row.lower_prime)
            assert isprime(row.upper_prime)
            # Check no prime between lower and upper
            for p in range(row.lower_prime + 1, row.upper_prime):
                assert not isprime(p), (
                    f"prime {p} between {row.lower_prime} and {row.upper_prime}"
                )

    def test_lower_endpoint_in_interval(self) -> None:
        """Every row's lower endpoint must be in [L, U]."""
        lo, hi = 10, 50
        result = compute_prime_gap_profile(
            PrimeGapProfileRequest(lower_bound=lo, upper_bound=hi)
        )
        for row in result.rows:
            assert lo <= row.lower_prime <= hi

    def test_concatenation_consistency(self) -> None:
        """Splitting at a prime boundary preserves all rows exactly once."""
        r1 = compute_prime_gap_profile(
            PrimeGapProfileRequest(lower_bound=2, upper_bound=20)
        )
        r2 = compute_prime_gap_profile(
            PrimeGapProfileRequest(lower_bound=23, upper_bound=50)
        )
        r_union = compute_prime_gap_profile(
            PrimeGapProfileRequest(lower_bound=2, upper_bound=50)
        )
        # The union of r1 and r2 rows should equal r_union rows
        all_rows = list(r1.rows) + list(r2.rows)
        assert len(all_rows) == len(r_union.rows)
        for i, row in enumerate(all_rows):
            assert row.lower_prime == r_union.rows[i].lower_prime
            assert row.upper_prime == r_union.rows[i].upper_prime
            assert row.gap == r_union.rows[i].gap


class TestAdditionalArithmeticProfiles:
    def test_least_prime_factor(self) -> None:
        result = compute_least_prime_factor_profile(
            LeastPrimeFactorProfileRequest(lower_bound=1, upper_bound=10)
        )
        assert [row.least_prime_factor for row in result.rows] == [
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

    def test_euler_totient(self) -> None:
        result = compute_euler_totient_profile(
            EulerTotientProfileRequest(lower_bound=1, upper_bound=10)
        )
        assert [row.euler_totient for row in result.rows] == [
            1,
            1,
            2,
            2,
            4,
            2,
            6,
            4,
            6,
            4,
        ]

    def test_divisor_sum(self) -> None:
        result = compute_divisor_sum_profile(
            DivisorSumProfileRequest(lower_bound=1, upper_bound=6)
        )
        assert [row.divisor_sum for row in result.rows] == [1, 3, 4, 7, 6, 12]


def test_dense_additional_profiles_use_the_shared_row_request() -> None:
    request = IntervalProfileRowsRequest(lower_bound=1, upper_bound=10)
    assert request.width() == 10
