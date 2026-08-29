"""Tests for multiplier divisor-sum fibers and p-adic interval profiles."""

import pytest

from jacobian.canonical import CanonicalLimits, canonicalize_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._models import MAX_INTEGER_DIGITS
from jacobian.math.number_theory._preimage_models import (
    MAX_INTERVAL_PROFILE_RESULT_BYTES,
    MAX_INTERVAL_PROFILE_ROWS,
    MAX_INTERVAL_PROFILE_WORK,
    KSigmaPreimageRequest,
    PAdicIntervalProfileRequest,
)
from jacobian.math.number_theory._preimage_ops import (
    PREIMAGE_OPERATIONS,
    compute_ksigma_preimage,
    compute_p_adic_interval_profile,
)
from jacobian.math.number_theory.operations import (
    ksigma_preimage as native_ksigma_preimage,
)
from jacobian.math.number_theory.operations import (
    p_adic_interval_profile as native_p_adic_interval_profile,
)


def test_native_preimage_operations_accept_canonical_values() -> None:
    sigma_result = native_ksigma_preimage(2, 8)
    profile_result = native_p_adic_interval_profile(0, 10, 2)
    assert sigma_result.preimages == ("3",)
    assert profile_result.total_valuation == "8"


def test_ksigma_preimage_preserves_multiplier_contract() -> None:
    result = compute_ksigma_preimage(KSigmaPreimageRequest(k=2, target_value="8"))
    assert result.k == 2
    assert result.target_value == "8"
    assert result.preimages == ("3",)
    assert result.count == 1


def test_ksigma_preimage_nondivisible_target_has_empty_fiber() -> None:
    result = compute_ksigma_preimage(KSigmaPreimageRequest(k=3, target_value="8"))
    assert result.preimages == ()


def test_ksigma_preimage_search_boundary_is_complete() -> None:
    request = KSigmaPreimageRequest(k=100, target_value="10000000")
    result = compute_ksigma_preimage(request)
    assert all(
        100 * __import__("sympy").divisor_sigma(int(n)) == 10000000
        for n in result.preimages
    )


def test_ksigma_preimage_rejects_oversized_search() -> None:
    request = KSigmaPreimageRequest(k=1, target_value="100001")
    with pytest.raises(OperationDomainValidationError) as error:
        compute_ksigma_preimage(request)
    assert error.value.errors()[0]["type"] == "number_theory.ksigma_search_bound"


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


def test_p_adic_profile_admits_large_endpoint_when_exact_result_is_small() -> None:
    result = compute_p_adic_interval_profile(
        PAdicIntervalProfileRequest(start="9" * 256, length="1", prime="2")
    )
    assert [(row.valuation, row.count) for row in result.rows] == [(256, "1")]
    assert result.total_valuation == "256"
    assert len(canonicalize_json(result.model_dump(mode="json"))) <= (
        CanonicalLimits().max_output_bytes
    )


def test_p_adic_profile_admits_large_length_from_exact_valuation_sum() -> None:
    length = 10**252
    result = compute_p_adic_interval_profile(
        PAdicIntervalProfileRequest(
            start="0",
            length="1" + "0" * 252,
            prime="2",
        )
    )
    assert int(result.total_valuation) == length - length.bit_count()
    assert len(result.total_valuation) <= MAX_INTEGER_DIGITS
    assert len(result.rows) <= MAX_INTERVAL_PROFILE_ROWS


def test_p_adic_profile_schema_exposes_coupled_endpoint_admission() -> None:
    schema = PAdicIntervalProfileRequest.model_json_schema()
    assert "start + length" in schema["description"]
    assert "decimal_digits(prime)^3" in schema["description"]
    assert schema["endpoint_sum_admission"] == {
        "endpoint": "start + length",
        "max_profile_powers": MAX_INTERVAL_PROFILE_ROWS,
        "max_profile_work_units": MAX_INTERVAL_PROFILE_WORK,
        "primality_work_units": "decimal_digits(prime)^3",
        "total_valuation_max_digits": MAX_INTEGER_DIGITS,
        "canonical_result_max_bytes": MAX_INTERVAL_PROFILE_RESULT_BYTES,
    }
    assert "start + length" in schema["properties"]["length"]["description"]

    operation = next(
        operation
        for operation in PREIMAGE_OPERATIONS
        if operation.operation_id
        == "number_theory.integer_interval.p_adic_valuation_profile.compute"
    )
    assert "coupled endpoint" in operation.examples[0].description
    assert "admission envelope" in operation.examples[0].description


def test_p_adic_profile_rejects_primality_work_before_backend_execution() -> None:
    large_prime = str(2**521 - 1)
    request = PAdicIntervalProfileRequest(start="0", length="1", prime=large_prime)

    with pytest.raises(OperationDomainValidationError) as error:
        compute_p_adic_interval_profile(request)

    assert error.value.errors()[0]["type"] == (
        "number_theory.p_adic_interval_profile_row_bound"
    )
    assert "including primality testing" in str(error.value)


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
