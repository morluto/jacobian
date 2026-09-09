from __future__ import annotations

from math import log

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.number_theory._dickman_rho import (
    DickmanRhoPiecewiseEnclosureRequest,
    DickmanRhoPiecewiseEnclosureResult,
    dickman_rho_piecewise_enclosure,
)


def _request(endpoint: int, exponent: int) -> DickmanRhoPiecewiseEnclosureRequest:
    return DickmanRhoPiecewiseEnclosureRequest.model_validate(
        {
            "endpoint": {"num": endpoint, "den": 1},
            "target_width": {"mantissa": 1, "exponent": exponent},
        }
    )


def _evaluate_bounds(
    result: DickmanRhoPiecewiseEnclosureResult, u: float
) -> tuple[float, float]:
    piece = next(
        piece
        for piece in result.pieces
        if float(piece.lower.as_fraction()) <= u <= float(piece.upper.as_fraction())
    )
    x = u - float(piece.lower.as_fraction())
    lower = sum(
        float(ball.lower.as_fraction()) * x**k
        for k, ball in enumerate(piece.coefficients)
    )
    upper = sum(
        float(ball.upper.as_fraction()) * x**k
        for k, ball in enumerate(piece.coefficients)
    )
    remainder = float(piece.uniform_remainder.as_fraction())
    return lower - remainder, upper + remainder


def test_rho_is_exact_on_first_unit_interval() -> None:
    result = dickman_rho_piecewise_enclosure(_request(1, -100))

    assert len(result.pieces) == 1
    for u in (0.0, 0.25, 1.0):
        assert _evaluate_bounds(result, u) == (1.0, 1.0)


def test_second_piece_encloses_one_minus_log_and_round_trips() -> None:
    result = dickman_rho_piecewise_enclosure(_request(2, -5))

    for u in (1.0, 1.25, 1.5, 1.75, 2.0):
        lower, upper = _evaluate_bounds(result, u)
        assert lower <= 1 - log(u) <= upper
    assert (
        DickmanRhoPiecewiseEnclosureResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_refinement_reduces_the_piece_width() -> None:
    coarse = dickman_rho_piecewise_enclosure(_request(2, -4))
    fine = dickman_rho_piecewise_enclosure(_request(2, -6))

    assert fine.degree > coarse.degree
    assert (
        fine.pieces[-1].uniform_remainder.compare(coarse.pieces[-1].uniform_remainder)
        < 0
    )


def test_unattainable_width_is_rejected_before_publication() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        dickman_rho_piecewise_enclosure(_request(8, -12))


def test_result_rejects_a_forged_remainder() -> None:
    result = dickman_rho_piecewise_enclosure(_request(2, -5))
    payload = result.model_dump()
    payload["pieces"][-1]["uniform_remainder"] = {"mantissa": 0, "exponent": 0}

    with pytest.raises(ValidationError, match="proved residual"):
        DickmanRhoPiecewiseEnclosureResult.model_validate(payload)
