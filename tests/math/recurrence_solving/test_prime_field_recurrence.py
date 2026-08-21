"""Tests for Berlekamp-Massey recurrence finding over prime fields."""

import pytest

from jacobian.math.recurrence_solving._models import (
    PrimeFieldRecurrenceFindRequest,
)
from jacobian.math.recurrence_solving._operations import (
    compute_prime_field_find_recurrence,
)


def _verify_recurrence(sequence, coefficients, prime):
    order = len(coefficients)
    for n in range(order, len(sequence)):
        expected = (
            sum(coefficients[i] * sequence[n - 1 - i] for i in range(order)) % prime
        )
        assert sequence[n] == expected, (n, sequence[n], expected)


class TestPrimeFieldRecurrenceFind:
    def test_fibonacci_mod_7(self):
        fib = [0, 1, 1, 2, 3, 5, 1, 6, 0]
        result = compute_prime_field_find_recurrence(
            PrimeFieldRecurrenceFindRequest(prime=7, sequence=tuple(fib)),
        )
        assert result.status == "FOUND"
        assert result.order == 2
        assert tuple(result.coefficients) == (1, 1)
        _verify_recurrence(fib, result.coefficients, 7)

    def test_geometric_sequence(self):
        # 1,2,4,1,2,4 mod 7 -> s_n = 2 s_{n-1}.
        geo = [1, 2, 4, 1, 2, 4]
        result = compute_prime_field_find_recurrence(
            PrimeFieldRecurrenceFindRequest(prime=7, sequence=tuple(geo)),
        )
        assert result.status == "FOUND"
        assert result.order == 1
        assert tuple(result.coefficients) == (2,)
        _verify_recurrence(geo, result.coefficients, 7)

    def test_all_zeros_no_recurrence(self):
        result = compute_prime_field_find_recurrence(
            PrimeFieldRecurrenceFindRequest(prime=5, sequence=(0, 0, 0, 0)),
        )
        assert result.status == "FOUND"
        assert result.order == 0
        assert result.coefficients == ()
        assert result.sequence == (0, 0, 0, 0)

    def test_constant_sequence(self):
        # 3,3,3,3 mod 5 -> s_n = s_{n-1}, coefficient (1,).
        result = compute_prime_field_find_recurrence(
            PrimeFieldRecurrenceFindRequest(prime=5, sequence=(3, 3, 3, 3)),
        )
        assert result.status == "FOUND"
        assert result.order == 1
        assert tuple(result.coefficients) == (1,)

    def test_minimality(self):
        # A sequence satisfying a length-1 recurrence should not get a longer one.
        seq = [1, 3, 2, 6, 4, 5, 1]  # mod 7: multiply by 3 each time
        result = compute_prime_field_find_recurrence(
            PrimeFieldRecurrenceFindRequest(prime=7, sequence=tuple(seq)),
        )
        assert result.status == "FOUND"
        assert result.order == 1

    def test_rejects_nonprime(self):
        with pytest.raises(ValueError, match="prime"):
            PrimeFieldRecurrenceFindRequest(prime=4, sequence=(1, 2, 3))

    def test_rejects_out_of_range_value(self):
        with pytest.raises(ValueError, match="residues"):
            PrimeFieldRecurrenceFindRequest(prime=3, sequence=(1, 3, 2))
