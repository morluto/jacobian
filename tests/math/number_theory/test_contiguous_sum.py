"""Tests for contiguous-sum representation profiles."""

import pytest
from pydantic import ValidationError

from jacobian.math.number_theory._contiguous_sum import CONTIGUOUS_SUM_OPERATION
from jacobian.math.number_theory._contiguous_sum_models import (
    ContiguousSumProfileRequest,
    ContiguousSumProfileResult,
)
from jacobian.math.number_theory._contiguous_sum_operations import (
    compute_contiguous_sum_profile,
)


def test_small() -> None:
    result = compute_contiguous_sum_profile(
        ContiguousSumProfileRequest(lower_bound="1", upper_bound="15")
    )
    counts = [r.representation_count for r in result.rows]
    # 15 = 15, 7+8, 4+5+6, 1+2+3+4+5 -> 4
    assert counts[14] == 4
    # 9 = 9, 4+5, 2+3+4 -> 3
    assert counts[8] == 3


def test_primes() -> None:
    """Primes have exactly 2 representations (as n and as sum of consecutive ints from 1 to some point)."""
    result = compute_contiguous_sum_profile(
        ContiguousSumProfileRequest(lower_bound="1", upper_bound="50")
    )
    from sympy import isprime

    for row in result.rows:
        if isprime(int(row.n)) and int(row.n) > 2:
            # Primes > 2 have exactly 1 odd divisor > 1 (themselves, odd), so 2 representations
            assert row.representation_count == 2, (
                f"n={row.n} has {row.representation_count} representations"
            )


@pytest.mark.parametrize("value", [True, 1000001, "01"])
def test_request_endpoints_are_strict_integers(value: object) -> None:
    with pytest.raises(ValidationError):
        ContiguousSumProfileRequest.model_validate(
            {"lower_bound": value, "upper_bound": value}
        )


def test_high_magnitude_singleton_does_not_allocate_to_upper_bound() -> None:
    result = compute_contiguous_sum_profile(
        ContiguousSumProfileRequest(
            lower_bound="1099511627776",
            upper_bound="1099511627776",
        )
    )
    assert result.rows[0].representation_count == 1


def test_large_endpoint_uses_canonical_strings_and_immutable_rows() -> None:
    result = compute_contiguous_sum_profile(
        ContiguousSumProfileRequest(
            lower_bound="9007199254740992",
            upper_bound="9007199254740992",
        )
    )

    assert result.lower_bound == "9007199254740992"
    assert result.upper_bound == "9007199254740992"
    assert result.rows[0].n == "9007199254740992"
    assert isinstance(result.rows, tuple)
    with pytest.raises(AttributeError):
        result.rows.append(result.rows[0])  # type: ignore[attr-defined]


def test_unknown_profile_rejects_rows() -> None:
    with pytest.raises(ValidationError):
        ContiguousSumProfileResult.model_validate(
            {
                "status": "UNKNOWN",
                "lower_bound": "1000000000001",
                "upper_bound": "1000000000001",
                "rows": [{"n": "1000000000001", "representation_count": 1}],
                "detail": "worker did not finish",
            }
        )


def test_request_schema_publishes_coupled_bounds() -> None:
    schema = ContiguousSumProfileRequest.model_json_schema()
    description = schema["description"]
    bounds = schema["x-jacobian-bounds"]

    assert "100,000" in description
    assert "direct factorization" in description
    assert schema["properties"]["lower_bound"]["type"] == "string"
    assert bounds["max_interval_width"] == 100_000
    assert CONTIGUOUS_SUM_OPERATION.examples[0].description.endswith(
        "interval must contain at most 100,000 integers."
    )
