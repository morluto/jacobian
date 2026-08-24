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
