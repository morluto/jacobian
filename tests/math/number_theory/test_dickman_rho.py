from __future__ import annotations

from fractions import Fraction
from math import log

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math import number_theory
from jacobian.math.analysis._models import MAX_DYADIC_EXPONENT, ExactDyadic
from jacobian.math.number_theory._dickman_rho import (
    DickmanRhoAffinePiece,
    DickmanRhoPiecewiseEnclosureRequest,
    DickmanRhoPiecewiseEnclosureResult,
    DyadicCoefficientBall,
    dickman_rho_piecewise_enclosure,
)


def _request(
    endpoint: int,
    exponent: int,
    *,
    precision_bits: int = 128,
) -> DickmanRhoPiecewiseEnclosureRequest:
    return DickmanRhoPiecewiseEnclosureRequest.model_validate(
        {
            "endpoint": {"num": endpoint, "den": 1},
            "target_width": {"mantissa": 1, "exponent": exponent},
            "precision_bits": precision_bits,
        }
    )


def _evaluate_bounds(
    result: DickmanRhoPiecewiseEnclosureResult, u: float
) -> tuple[float, float]:
    piece = next(piece for piece in result.pieces if piece.lower <= u <= piece.upper)
    t = 2 * (u - float(piece.axis.center.as_fraction()))
    lower = sum(
        float(ball.lower.as_fraction()) * t**k
        for k, ball in enumerate(piece.coefficients)
    )
    upper = sum(
        float(ball.upper.as_fraction()) * t**k
        for k, ball in enumerate(piece.coefficients)
    )
    remainder = float(piece.uniform_remainder.as_fraction())
    return lower - remainder, upper + remainder


def _compute(
    request: DickmanRhoPiecewiseEnclosureRequest,
) -> DickmanRhoPiecewiseEnclosureResult:
    return dickman_rho_piecewise_enclosure(
        request.endpoint,
        request.target_width,
        precision_bits=request.precision_bits,
    )


def test_rho_is_exact_on_first_unit_interval() -> None:
    result = _compute(_request(1, -100))

    assert len(result.pieces) == 1
    assert (result.pieces[0].lower, result.pieces[0].upper) == (0, 1)
    assert result.pieces[0].axis.center.as_fraction() == 1 / 2
    assert result.pieces[0].axis.scale.as_fraction() == 1 / 2
    for u in (0.0, 0.25, 1.0):
        assert _evaluate_bounds(result, u) == (1.0, 1.0)


def test_second_piece_encloses_one_minus_log_and_round_trips() -> None:
    result = _compute(_request(2, -5))

    for u in (1.0, 1.25, 1.5, 1.75, 2.0):
        lower, upper = _evaluate_bounds(result, u)
        assert lower <= 1 - log(u) <= upper
    assert (
        DickmanRhoPiecewiseEnclosureResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_later_piece_overlaps_independent_high_precision_values() -> None:
    result = _compute(_request(3, -40))

    for u, expected in (
        (2.5, 0.130319561832251),
        (3.0, 0.0486083882911316),
    ):
        lower, upper = _evaluate_bounds(result, u)
        assert lower <= expected <= upper


def test_refinement_reduces_the_piece_width() -> None:
    coarse = _compute(_request(2, -20))
    fine = _compute(_request(2, -40))

    assert fine.degree > coarse.degree
    assert (
        fine.pieces[-1].uniform_remainder.compare(coarse.pieces[-1].uniform_remainder)
        < 0
    )


def test_unattainable_width_is_rejected_before_publication() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        _compute(_request(8, -140))


def test_width_below_the_degree_envelope_is_a_typed_resource_failure() -> None:
    with pytest.raises(
        OperationResourceAdmissionError,
        match="degree-256 enclosure",
    ):
        _compute(_request(2, -500, precision_bits=512))


def test_result_rejects_a_structurally_excessive_remainder() -> None:
    result = _compute(_request(2, -5))
    payload = result.model_dump()
    payload["pieces"][-1]["uniform_remainder"] = {"mantissa": 1, "exponent": 0}

    with pytest.raises(ValidationError, match="requested uniform width"):
        DickmanRhoPiecewiseEnclosureResult.model_validate(payload)


def test_fractional_endpoint_returns_complete_integer_partition() -> None:
    result = _compute(
        DickmanRhoPiecewiseEnclosureRequest.model_validate(
            {
                "endpoint": {"num": 5, "den": 2},
                "target_width": {"mantissa": 1, "exponent": -20},
            }
        )
    )

    assert [(piece.lower, piece.upper) for piece in result.pieces] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]


def test_precision_is_bounded_and_controls_narrowing() -> None:
    coarse = _compute(_request(2, -50, precision_bits=64))
    fine = _compute(_request(2, -50, precision_bits=256))

    coarse_width = coarse.pieces[-1].uniform_remainder.as_fraction()
    fine_width = fine.pieces[-1].uniform_remainder.as_fraction()
    assert fine_width < coarse_width
    with pytest.raises(ValidationError):
        _request(2, -20, precision_bits=16)


def test_exact_initial_interval_accepts_width_below_precision_floor() -> None:
    result = _compute(_request(1, -200, precision_bits=32))

    assert result.pieces[0].uniform_remainder.as_fraction() == 0


def test_centered_recurrence_has_zero_low_order_residual() -> None:
    result = _compute(_request(3, -20))
    previous = result.pieces[0]
    current = result.pieces[1]

    def interval(ball: DyadicCoefficientBall) -> tuple[Fraction, Fraction]:
        return ball.lower.as_fraction(), ball.upper.as_fraction()

    for k in range(result.degree):
        previous_interval = interval(previous.coefficients[k])
        current_k = interval(current.coefficients[k])
        current_next = interval(current.coefficients[k + 1])
        low = 3 * (k + 1) * current_next[0] + k * current_k[0] + previous_interval[0]
        high = 3 * (k + 1) * current_next[1] + k * current_k[1] + previous_interval[1]
        assert low <= 0 <= high


def test_fine_enclosure_certifies_positivity_and_monotonicity() -> None:
    result = _compute(_request(3, -40))

    def evaluate(
        piece: DickmanRhoAffinePiece, t: Fraction, derivative: bool = False
    ) -> tuple[Fraction, Fraction]:
        lower = Fraction()
        upper = Fraction()
        for k, coefficient in enumerate(piece.coefficients):
            if derivative and k == 0:
                continue
            multiplier = k * t ** (k - 1) if derivative else t**k
            values = (
                coefficient.lower.as_fraction(),
                coefficient.upper.as_fraction(),
            )
            terms = tuple(multiplier * value for value in values)
            lower += min(terms)
            upper += max(terms)
        remainder = Fraction() if derivative else piece.uniform_remainder.as_fraction()
        return lower - remainder, upper + remainder

    for piece in result.pieces:
        for numerator in range(-4, 5):
            t = Fraction(numerator, 4)
            lower, _ = evaluate(piece, t)
            _, derivative_upper = evaluate(piece, t, derivative=True)
            assert lower >= 0
            assert derivative_upper <= 0


def test_dickman_native_api_exports_canonical_value_family() -> None:
    expected = {
        "DickmanRhoAffineAxis",
        "DickmanRhoAffinePiece",
        "DickmanRhoPiecewiseEnclosureParameters",
        "DickmanRhoPiecewiseEnclosureRequest",
        "DickmanRhoPiecewiseEnclosureResult",
        "DyadicCoefficientBall",
        "dickman_rho_piecewise_enclosure",
    }
    assert expected <= set(number_theory.__all__)
    assert all(hasattr(number_theory, name) for name in expected)


def test_native_endpoint_outside_the_contract_is_a_typed_domain_error() -> None:
    from jacobian._exact import CanonicalRational

    with pytest.raises(OperationDomainValidationError, match=r"\[0, 8\]"):
        dickman_rho_piecewise_enclosure(
            CanonicalRational(num=9, den=1),
            ExactDyadic(mantissa=1, exponent=-5),
        )


def test_native_noncanonical_argument_types_are_rejected() -> None:

    with pytest.raises(OperationDomainValidationError, match="CanonicalRational"):
        dickman_rho_piecewise_enclosure(
            Fraction(2),  # type: ignore[arg-type]
            ExactDyadic(mantissa=1, exponent=-5),
        )


def test_endpoint_eight_work_estimate_is_admitted() -> None:
    result = dickman_rho_piecewise_enclosure(
        CanonicalRational(num=8, den=1),
        ExactDyadic(mantissa=1, exponent=-5),
        precision_bits=32,
    )
    assert result.pieces
    assert result.pieces[0].lower == 0
    assert result.pieces[-1].upper == 8


def test_extreme_dyadic_target_width_is_rejected_without_fraction_expansion() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="exponent"):
        dickman_rho_piecewise_enclosure(
            CanonicalRational(num=2, den=1),
            ExactDyadic(mantissa=1, exponent=-(MAX_DYADIC_EXPONENT)),
        )
    with pytest.raises(OperationResourceAdmissionError, match="exponent"):
        dickman_rho_piecewise_enclosure(
            CanonicalRational(num=2, den=1),
            ExactDyadic(mantissa=1, exponent=MAX_DYADIC_EXPONENT),
        )


def test_precision_bits_must_be_a_strict_int() -> None:
    with pytest.raises(OperationDomainValidationError, match="int"):
        dickman_rho_piecewise_enclosure(
            CanonicalRational(num=2, den=1),
            ExactDyadic(mantissa=1, exponent=-5),
            precision_bits="128",  # type: ignore[arg-type]
        )
    with pytest.raises(OperationDomainValidationError, match="int"):
        dickman_rho_piecewise_enclosure(
            CanonicalRational(num=2, den=1),
            ExactDyadic(mantissa=1, exponent=-5),
            precision_bits=128.0,  # type: ignore[arg-type]
        )
