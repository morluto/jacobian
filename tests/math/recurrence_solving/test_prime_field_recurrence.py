"""Tests for Berlekamp-Massey recurrence finding over prime fields."""

import pytest
from pydantic import ValidationError

from jacobian.math.recurrence_solving._models import (
    PrimeFieldRecurrenceFindRequest,
    PrimeFieldRecurrenceFindResult,
)
from jacobian.math.recurrence_solving._operations import (
    compute_prime_field_find_recurrence,
)

FIBONACCI_MOD_7 = (0, 1, 1, 2, 3, 5, 1, 6, 0)


def _verify_recurrence(sequence, coefficients, prime):
    order = len(coefficients)
    for n in range(order, len(sequence)):
        expected = (
            sum(coefficients[i] * sequence[n - 1 - i] for i in range(order)) % prime
        )
        assert sequence[n] == expected, (n, sequence[n], expected)


class TestPrimeFieldRecurrenceFind:
    def test_fibonacci_mod_7(self):
        result = compute_prime_field_find_recurrence(
            PrimeFieldRecurrenceFindRequest(prime=7, sequence=FIBONACCI_MOD_7),
        )
        assert result.status == "FOUND"
        assert result.order == 2
        assert tuple(result.coefficients) == (1, 1)
        _verify_recurrence(FIBONACCI_MOD_7, result.coefficients, 7)

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

    def test_request_schema_publishes_canonical_residue_constraint(self):
        schema = PrimeFieldRecurrenceFindRequest.model_json_schema()
        assert "0 <= value < prime" in schema["properties"]["sequence"]["description"]
        assert "2" in schema["properties"]["prime"]["description"]

    def test_result_schema_publishes_canonical_residue_constraint(self):
        schema = PrimeFieldRecurrenceFindResult.model_json_schema()
        assert "0 <= value < prime" in schema["properties"]["sequence"]["description"]
        assert (
            "0 <= value < prime" in schema["properties"]["coefficients"]["description"]
        )

    def test_result_accepts_minimal_fibonacci_recurrence(self):
        result = PrimeFieldRecurrenceFindResult(
            prime=7,
            sequence=FIBONACCI_MOD_7,
            coefficients=(1, 1),
            order=2,
            status="FOUND",
        )
        assert result.status == "FOUND"
        assert result.order == 2
        assert result.coefficients == (1, 1)

    def test_result_rejects_non_minimal_fibonacci_recurrence(self):
        with pytest.raises(ValidationError, match="Berlekamp-Massey"):
            PrimeFieldRecurrenceFindResult(
                prime=7,
                sequence=FIBONACCI_MOD_7,
                coefficients=(1, 1, 0),
                order=3,
                status="FOUND",
            )

    def test_result_rejects_false_no_fitting_recurrence(self):
        with pytest.raises(ValidationError, match="Berlekamp-Massey"):
            PrimeFieldRecurrenceFindResult(
                prime=7,
                sequence=FIBONACCI_MOD_7,
                coefficients=(),
                order=0,
                status="NO_FITTING_RECURRENCE",
            )

    def test_result_rejects_wrong_coefficients_at_minimal_order(self):
        with pytest.raises(ValidationError, match="Berlekamp-Massey"):
            PrimeFieldRecurrenceFindResult(
                prime=7,
                sequence=FIBONACCI_MOD_7,
                coefficients=(1, 2),
                order=2,
                status="FOUND",
            )

    def test_impulse_sequence_is_the_minimal_order_n_recurrence(self):
        sequence = (0, 0, 0, 1)
        result = compute_prime_field_find_recurrence(
            PrimeFieldRecurrenceFindRequest(prime=2, sequence=sequence),
        )
        assert result.status == "FOUND"
        assert result.order == len(sequence)
        PrimeFieldRecurrenceFindResult.model_validate(result.model_dump())
        with pytest.raises(ValidationError, match="Berlekamp-Massey"):
            PrimeFieldRecurrenceFindResult(
                prime=2,
                sequence=sequence,
                coefficients=(),
                order=0,
                status="NO_FITTING_RECURRENCE",
            )
