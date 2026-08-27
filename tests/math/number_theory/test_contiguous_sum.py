"""Tests for contiguous-sum representation profiles."""

from jacobian.math.number_theory._contiguous_sum_models import (
    ContiguousSumProfileRequest,
)
from jacobian.math.number_theory._contiguous_sum_operations import (
    compute_contiguous_sum_profile,
)


def test_small() -> None:
    result = compute_contiguous_sum_profile(
        ContiguousSumProfileRequest(lower_bound=1, upper_bound=15)
    )
    counts = [r.representation_count for r in result.rows]
    # 15 = 15, 7+8, 4+5+6, 1+2+3+4+5 -> 4
    assert counts[14] == 4
    # 9 = 9, 4+5, 2+3+4 -> 3
    assert counts[8] == 3


def test_primes() -> None:
    """Primes have exactly 2 representations (as n and as sum of consecutive ints from 1 to some point)."""
    result = compute_contiguous_sum_profile(
        ContiguousSumProfileRequest(lower_bound=1, upper_bound=50)
    )
    from sympy import isprime

    for row in result.rows:
        if isprime(row.n) and row.n > 2:
            # Primes > 2 have exactly 1 odd divisor > 1 (themselves, odd), so 2 representations
            assert row.representation_count == 2, (
                f"n={row.n} has {row.representation_count} representations"
            )
