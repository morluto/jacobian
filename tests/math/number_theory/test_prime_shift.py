"""Tests for translated-prime representation profiles."""

from jacobian.math.number_theory._prime_shift_models import PrimeShiftProfileRequest
from jacobian.math.number_theory._prime_shift_operations import compute_prime_shift_profile


def test_basic_interval() -> None:
    result = compute_prime_shift_profile(PrimeShiftProfileRequest(lower_bound=1, upper_bound=20))
    counts = [r.representation_count for r in result.rows]
    # n=4: 4=2+2, 4=3+1 -> 2 representations
    # n=7: 7=5+2, 7=3+4, 7=2+5? no, 7-2=5 (prime), 7-4=3 (prime), 7-8<0
    # Let's verify n=4: p + 2^k = 4, p prime, k>=0: 4-1=3(prime), 4-2=2(prime), 4-4=0(not prime), 4-8<0
    # So 4 has 2 representations: 3+1, 2+2
    assert counts[3] == 2  # n=4 has 2 representations


def test_small_values() -> None:
    result = compute_prime_shift_profile(PrimeShiftProfileRequest(lower_bound=1, upper_bound=5))
    counts = [r.representation_count for r in result.rows]
    # n=1: 1-1=0(not prime) -> 0
    # n=2: 2-1=1(not prime), 2-2=0(not prime) -> 0
    # n=3: 3-1=2(prime), 3-2=1(not prime), 3-4<0 -> 1
    # n=4: 4-1=3(prime), 4-2=2(prime), 4-4=0(not prime) -> 2
    # n=5: 5-1=4(not prime), 5-2=3(prime), 5-4=1(not prime) -> 1
    assert counts == [0, 0, 1, 2, 1]


def test_single_element() -> None:
    result = compute_prime_shift_profile(PrimeShiftProfileRequest(lower_bound=4, upper_bound=4))
    assert len(result.rows) == 1
    assert result.rows[0].n == 4
    assert result.rows[0].representation_count == 2
