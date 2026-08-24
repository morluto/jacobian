from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.math.markov_chain._models import MixingTimeRequest
from jacobian.math.markov_chain._operations import compute_mixing_time


def _r(num: int, den: int = 1) -> dict[str, str]:
    return {"num": str(num), "den": str(den)}


def _request(
    *, epsilon: tuple[int, int] = (1, 100), max_steps: int = 8
) -> MixingTimeRequest:
    return MixingTimeRequest.model_validate(
        {
            "matrix": [[_r(1, 2), _r(1, 2)], [_r(1, 4), _r(3, 4)]],
            "epsilon": _r(*epsilon),
            "max_steps": max_steps,
        }
    )


def test_exact_two_state_mixing_time_and_distance() -> None:
    result = compute_mixing_time(_request())

    assert result.status == "FOUND"
    assert result.mixing_time == 4
    assert result.max_total_variation_distance is not None
    assert result.max_total_variation_distance.as_fraction() == Fraction(1, 384)
    assert result.steps_examined == 5


def test_search_checks_time_zero_and_boundary_equality() -> None:
    time_zero = compute_mixing_time(_request(epsilon=(1, 1)))
    assert time_zero.mixing_time == 0
    assert time_zero.steps_examined == 1
    equality = compute_mixing_time(_request(epsilon=(1, 6)))
    assert equality.mixing_time == 1
    assert equality.max_total_variation_distance is not None
    assert equality.max_total_variation_distance.as_fraction() == Fraction(1, 6)
    assert equality.steps_examined == 2


def test_bound_exceeded_returns_terminal_exact_distance() -> None:
    result = compute_mixing_time(_request(max_steps=3))
    assert result.status == "BOUND_EXCEEDED"
    assert result.max_total_variation_distance is not None
    assert result.max_total_variation_distance.as_fraction() == Fraction(1, 96)
    assert result.steps_examined == 4


def test_found_mixing_time_is_the_first_satisfactory_step() -> None:
    result = compute_mixing_time(_request())
    assert result.mixing_time is not None and result.mixing_time > 0
    previous = compute_mixing_time(_request(max_steps=result.mixing_time - 1))
    assert previous.status == "BOUND_EXCEEDED"
    assert previous.max_total_variation_distance is not None
    assert previous.max_total_variation_distance.as_fraction() > Fraction(1, 100)


@pytest.mark.parametrize(
    "matrix", [[[_r(0), _r(1)], [_r(1), _r(0)]], [[_r(1), _r(0)], [_r(0), _r(1)]]]
)
def test_nonergodic_chains_return_typed_outcome(
    matrix: list[list[dict[str, str]]],
) -> None:
    request = MixingTimeRequest.model_validate(
        {"matrix": matrix, "epsilon": _r(1, 10), "max_steps": 4}
    )
    assert compute_mixing_time(request).status == "NOT_ERGODIC"


def test_search_bounds_reject_before_exact_matrix_powers() -> None:
    with pytest.raises(ValidationError, match="at most 32"):
        MixingTimeRequest.model_validate(
            {
                "matrix": [
                    [_r(1 if i == j else 0) for j in range(33)] for i in range(33)
                ],
                "epsilon": _r(1, 10),
                "max_steps": 4,
            }
        )
    with pytest.raises(ValidationError, match=r"\(0, 1\]"):
        _request(epsilon=(0, 1))


def test_search_rejects_a_rational_height_that_cannot_fit_the_result() -> None:
    denominator = 10**31
    matrix = [
        [
            _r(denominator - 1, denominator)
            if row == column
            else _r(1, denominator)
            if column == (row + 1) % 8
            else _r(0)
            for column in range(8)
        ]
        for row in range(8)
    ]
    with pytest.raises(ValidationError, match="exact rational result bound"):
        MixingTimeRequest.model_validate(
            {"matrix": matrix, "epsilon": _r(1, 10), "max_steps": 256}
        )
