"""Tests for translated-prime representation profiles."""

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory import PrimeShiftProfileResult, prime_shift_profile
from jacobian.math.number_theory._prime_shift_models import PrimeShiftProfileRequest
from jacobian.math.number_theory._prime_shifts import (
    compute_prime_shift_profile,
)


def test_basic_interval() -> None:
    result = compute_prime_shift_profile(
        PrimeShiftProfileRequest(lower_bound=1, upper_bound=20)
    )
    counts = [r.representation_count for r in result.rows]
    # n=4: 4=2+2, 4=3+1 -> 2 representations
    # n=7: 7=5+2, 7=3+4, 7=2+5? no, 7-2=5 (prime), 7-4=3 (prime), 7-8<0
    # Let's verify n=4: p + 2^k = 4, p prime, k>=0: 4-1=3(prime), 4-2=2(prime), 4-4=0(not prime), 4-8<0
    # So 4 has 2 representations: 3+1, 2+2
    assert counts[3] == 2  # n=4 has 2 representations


def test_small_values() -> None:
    result = compute_prime_shift_profile(
        PrimeShiftProfileRequest(lower_bound=1, upper_bound=5)
    )
    counts = [r.representation_count for r in result.rows]
    # n=1: 1-1=0(not prime) -> 0
    # n=2: 2-1=1(not prime), 2-2=0(not prime) -> 0
    # n=3: 3-1=2(prime), 3-2=1(not prime), 3-4<0 -> 1
    # n=4: 4-1=3(prime), 4-2=2(prime), 4-4=0(not prime) -> 2
    # n=5: 5-1=4(not prime), 5-2=3(prime), 5-4=1(not prime) -> 1
    assert counts == [0, 0, 1, 2, 1]


def test_native_api_matches_wire_operation() -> None:
    request = PrimeShiftProfileRequest(lower_bound=1, upper_bound=20)
    wire_result = compute_prime_shift_profile(request)
    native_result = prime_shift_profile(1, 20)

    assert isinstance(native_result, PrimeShiftProfileResult)
    assert native_result == wire_result


def test_single_element() -> None:
    result = compute_prime_shift_profile(
        PrimeShiftProfileRequest(lower_bound=4, upper_bound=4)
    )
    assert len(result.rows) == 1
    assert result.rows[0].n == 4
    assert result.rows[0].representation_count == 2


def test_admits_narrow_interval_above_legacy_upper_bound() -> None:
    result = compute_prime_shift_profile(
        PrimeShiftProfileRequest(lower_bound=10_000_001, upper_bound=10_000_001)
    )

    assert [(row.n, row.representation_count) for row in result.rows] == [
        (10_000_001, 2)
    ]


def test_admits_singleton_at_2_to_the_32() -> None:
    result = compute_prime_shift_profile(
        PrimeShiftProfileRequest(lower_bound=2**32, upper_bound=2**32)
    )

    assert len(result.rows) == 1
    assert result.rows[0].n == 2**32


def test_result_rows_are_immutable() -> None:
    result = compute_prime_shift_profile(
        PrimeShiftProfileRequest(lower_bound=4, upper_bound=4)
    )

    assert isinstance(result.rows, tuple)
    with pytest.raises(TypeError):
        result.rows[0] = result.rows[0]  # type: ignore[index]
    assert not hasattr(result.rows, "append")


@pytest.mark.parametrize(
    "payload",
    [
        {"lower_bound": 4, "upper_bound": 5, "rows": []},
        {
            "lower_bound": 4,
            "upper_bound": 5,
            "rows": [
                {"n": 4, "representation_count": 0},
                {"n": 4, "representation_count": 0},
            ],
        },
        {
            "lower_bound": 4,
            "upper_bound": 5,
            "rows": [
                {"n": 5, "representation_count": 0},
                {"n": 4, "representation_count": 0},
            ],
        },
        {
            "lower_bound": 4,
            "upper_bound": 5,
            "rows": [
                {"n": 4, "representation_count": 0},
                {"n": 6, "representation_count": 0},
            ],
        },
        {"lower_bound": 5, "upper_bound": 4, "rows": []},
    ],
    ids=["missing", "duplicated", "out-of-order", "outside", "invalid-bounds"],
)
def test_deserialized_result_binds_declared_row_axis(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match=r"declared interval|upper_bound"):
        PrimeShiftProfileResult.model_validate(payload)


def test_result_axis_round_trips_through_serialization() -> None:
    result = compute_prime_shift_profile(
        PrimeShiftProfileRequest(lower_bound=4, upper_bound=5)
    )

    restored = PrimeShiftProfileResult.model_validate(result.model_dump(mode="json"))

    assert tuple(row.n for row in restored.rows) == (4, 5)


def test_rejects_interval_that_exceeds_segmented_sieve_work_budget() -> None:
    request = PrimeShiftProfileRequest(
        lower_bound=4_000_000_000,
        upper_bound=4_000_200_000,
    )

    with pytest.raises(OperationDomainValidationError, match="work budget"):
        compute_prime_shift_profile(request)


def test_rejects_profile_that_exceeds_canonical_output_budget() -> None:
    with pytest.raises(ValueError, match="canonical output budget"):
        compute_prime_shift_profile(
            PrimeShiftProfileRequest(lower_bound=9_000_001, upper_bound=10_000_000)
        )
