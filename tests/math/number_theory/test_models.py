from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.number_theory._models import (
    ChineseRemainderRequest,
    FactorialValuationRequest,
    FactorizationRequest,
    ModularValueRequest,
    NonnegativeIntegerRequest,
    PositiveIntegerRequest,
    PowerfulNumberResult,
)


@pytest.mark.parametrize("residue", [-1, 3])
def test_chinese_remainder_rejects_noncanonical_residues(residue: int) -> None:
    with pytest.raises(ValidationError, match="canonical"):
        ChineseRemainderRequest(residues=(residue,), moduli=(3,))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"residues": [1, 2], "moduli": [3]}, "equal length"),
        ({"residues": [0], "moduli": [1]}, "between 2 and 10,000"),
        ({"residues": [0], "moduli": [10_001]}, "between 2 and 10,000"),
    ],
)
def test_chinese_remainder_rejects_invalid_system_bounds(
    payload: dict[str, list[int]],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ChineseRemainderRequest.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    (
        {
            "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
            "is_powerful": False,
            "factors": [{"prime": "2", "power": 3}],
            "violating_primes": [],
        },
        {
            "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
            "is_powerful": True,
            "factors": [{"prime": "2", "power": 1}],
            "violating_primes": ["2"],
        },
        {
            "semantics_version": "powerful-number.prime-exponents-at-least-two.v1",
            "is_powerful": False,
            "factors": [
                {"prime": "3", "power": 1},
                {"prime": "2", "power": 2},
            ],
            "violating_primes": ["3"],
        },
    ),
)
def test_powerful_number_result_rejects_inconsistent_or_noncanonical_witnesses(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match=r"powerful|factor"):
        PowerfulNumberResult.model_validate(payload)


def test_in_process_factorization_dependencies_have_small_input_bounds() -> None:
    for model, payload in (
        (PositiveIntegerRequest, {"n": 1_001}),
        (NonnegativeIntegerRequest, {"n": 1_001}),
        (ModularValueRequest, {"value": "2", "modulus": 10_001}),
        (FactorialValuationRequest, {"n": 1, "base": 1_000_001}),
        (FactorizationRequest, {"value": "1000000000000"}),
    ):
        with pytest.raises(ValidationError, match=r"less than or equal|at most"):
            model.model_validate(payload)
