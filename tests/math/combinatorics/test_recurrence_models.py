from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import pytest
from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from jacobian.math.combinatorics._recurrence_models import (
    MAX_COMBINATORICS_RESULT_ARTIFACT_BYTES,
    PolynomialCoefficientRecurrenceEvaluationRequest,
    PolynomialCoefficientRecurrenceEvaluationResult,
    _validate_result_inline_size,
)


@contextmanager
def raises_code(code: str):
    with pytest.raises(ValidationError) as exc_info:
        yield
    assert exc_info.value.errors()[0]["type"] == code


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
        [
            {"index": 0, "value": _q(1)},
            {"index": 2, "value": _q(2)},
            {"index": 1, "value": _q(1)},
            {"index": 3, "value": _q(3)},
        ],
        [
            {"index": 0, "value": _q(1)},
            {"index": 1, "value": _q(1)},
            {"index": 2, "value": _q(2)},
            {"index": 3, "value": _q(3)},
            {"index": 4, "value": _q(5)},
        ],
    ],
)
def test_polynomial_recurrence_result_rejects_malformed_prefix_projection(
    values: list[dict[str, object]],
) -> None:
    result = _result()
    result["values"] = values

    with raises_code("combinatorics.result_bound"):
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
        [
            {"index": 3, "value": _q(0)},
            {"index": 2, "value": _q(0)},
        ],
    ],
)
def test_polynomial_recurrence_result_requires_exact_residual_range(
    residuals: list[dict[str, object]],
) -> None:
    result = deepcopy(_result())
    result["residuals"] = residuals

    with raises_code("combinatorics.recurrence_invariant"):
        PolynomialCoefficientRecurrenceEvaluationResult.model_validate(result)


def test_polynomial_recurrence_result_accepts_complete_bound_replay() -> None:
    result = PolynomialCoefficientRecurrenceEvaluationResult.model_validate(_result())

    assert tuple(item.index for item in result.values) == (0, 1, 2, 3)
    assert tuple(item.index for item in result.residuals) == (2, 3)


def test_result_size_translation_preserves_owner_local_reason() -> None:
    with pytest.raises(PydanticCustomError) as caught:
        _validate_result_inline_size(
            {"payload": "x" * (MAX_COMBINATORICS_RESULT_ARTIFACT_BYTES + 1)}
        )

    assert caught.value.type == "combinatorics.result_bound"


def test_polynomial_recurrence_aborts_when_an_intermediate_exceeds_digit_bound() -> (
    None
):
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59)
    denominators = []
    for prime in primes:
        denominator = prime
        while len(str(denominator * prime)) <= 64:
            denominator *= prime
        denominators.append(str(denominator))
    request = {
        "coefficient_polynomials": [
            [_q(50), _q(-1)],
            [{"num": "1", "den": denominator} for denominator in denominators],
        ],
        "initial_values": [_q(1)],
        "coefficient_convention": (
            "SUM_P_J_OF_N_TIMES_A_N_MINUS_J_EQUALS_ZERO_FOR_J_FROM_0"
        ),
        "polynomial_convention": "ASCENDING_POWERS_OF_N",
        "scope": "PREFIX",
        "term_count": 100,
        "indices": [],
    }

    with raises_code("combinatorics.recurrence_invariant"):
        PolynomialCoefficientRecurrenceEvaluationRequest.model_validate(request)
