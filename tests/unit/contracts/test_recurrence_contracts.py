from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from jacobian.contracts.combinatorics import (
    PolynomialCoefficientRecurrenceEvaluationResult,
)


def _q(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


def _result() -> dict[str, object]:
    return {
        "coefficient_convention": (
            "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
        ),
        "polynomial_convention": "ASCENDING_POWERS_OF_N",
        "scope": "PREFIX",
        "recurrence_order": 2,
        "values": [
            {"index": index, "value": _q(value)}
            for index, value in enumerate((1, 1, 2, 3))
        ],
        "replay_prefix": [_q(value) for value in (1, 1, 2, 3)],
        "residuals": [{"index": index, "value": _q(0)} for index in range(2, 4)],
        "replay_scope_end": 3,
        "exactness": "EXACT_RATIONAL",
        "determinism": "DETERMINISTIC",
        "backend": "sympy",
        "backend_version": "1.14.0",
        "verification": "UNVERIFIED",
    }


@pytest.mark.parametrize(
    "values",
    [
        [
            {"index": 1, "value": _q(1)},
            {"index": 2, "value": _q(2)},
            {"index": 3, "value": _q(3)},
        ],
        [
            {"index": 0, "value": _q(1)},
            {"index": 2, "value": _q(2)},
            {"index": 2, "value": _q(2)},
            {"index": 3, "value": _q(3)},
        ],
    ],
)
def test_polynomial_recurrence_result_rejects_malformed_prefix_projection(
    values: list[dict[str, object]],
) -> None:
    result = _result()
    result["values"] = values

    with pytest.raises(ValidationError):
        PolynomialCoefficientRecurrenceEvaluationResult.model_validate(result)


@pytest.mark.parametrize(
    "residuals",
    [
        [],
        [{"index": 2, "value": _q(0)}],
        [
            {"index": 2, "value": _q(0)},
            {"index": 2, "value": _q(0)},
        ],
        [
            {"index": 2, "value": _q(0)},
            {"index": 4, "value": _q(0)},
        ],
    ],
)
def test_polynomial_recurrence_result_requires_exact_residual_range(
    residuals: list[dict[str, object]],
) -> None:
    result = deepcopy(_result())
    result["residuals"] = residuals

    with pytest.raises(ValidationError):
        PolynomialCoefficientRecurrenceEvaluationResult.model_validate(result)


def test_polynomial_recurrence_result_accepts_complete_bound_replay() -> None:
    result = PolynomialCoefficientRecurrenceEvaluationResult.model_validate(_result())

    assert tuple(item.index for item in result.values) == (0, 1, 2, 3)
    assert tuple(item.index for item in result.residuals) == (2, 3)
