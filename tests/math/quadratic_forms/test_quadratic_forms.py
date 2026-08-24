"""Exact and contract tests for rational quadratic-form evaluation."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.quadratic_forms import (
    RationalCoordinateVector,
    RationalQuadraticForm,
    evaluate_rational_quadratic_form,
)
from jacobian.math.quadratic_forms._models import EvaluationRequest, EvaluationResult
from jacobian.math.quadratic_forms._operations import evaluate_form
from jacobian.math.quadratic_forms.values import (
    MAX_QUADRATIC_EVALUATION_COMMON_DENOMINATOR_DIGITS,
    MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
    MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS,
)


def _rational(numerator: int, denominator: int = 1) -> dict[str, str]:
    return {"num": str(numerator), "den": str(denominator)}


def _form() -> dict[str, object]:
    return {
        "axis": ["x", "y"],
        "diagonal_coefficients": [_rational(2), _rational(5)],
        "cross_terms": [
            {"left": 0, "right": 1, "coefficient": _rational(3)},
        ],
    }


def _vector() -> dict[str, object]:
    return {
        "axis": ["x", "y"],
        "coordinates": [_rational(1, 2), _rational(2)],
    }


def test_evaluate_uses_explicit_polynomial_cross_term_convention() -> None:
    result = evaluate_form(
        EvaluationRequest.model_validate({"form": _form(), "vector": _vector()})
    )

    assert result.value == CanonicalRational(num="47", den="2")
    assert result.form.axis == ("x", "y")
    assert result.vector.axis == result.form.axis


def test_native_evaluation_returns_an_exact_fraction() -> None:
    form = RationalQuadraticForm.model_validate(_form())
    vector = RationalCoordinateVector.model_validate(_vector())

    assert evaluate_rational_quadratic_form(form, vector) == Fraction(47, 2)


def test_zero_form_is_a_complete_exact_value() -> None:
    form = {
        "axis": ["u"],
        "diagonal_coefficients": [_rational(0)],
    }
    vector = {"axis": ["u"], "coordinates": [_rational(-7, 3)]}

    assert evaluate_form(
        EvaluationRequest.model_validate({"form": form, "vector": vector})
    ).value == CanonicalRational(num="0", den="1")


def test_axis_mismatch_is_rejected_before_arithmetic() -> None:
    with pytest.raises(ValidationError, match="vector axis"):
        EvaluationRequest.model_validate(
            {
                "form": _form(),
                "vector": {
                    "axis": ["y", "x"],
                    "coordinates": [_rational(2), _rational(1)],
                },
            }
        )


def test_cross_terms_are_nonzero_unique_and_canonically_ordered() -> None:
    with pytest.raises(ValidationError, match="zero cross terms"):
        RationalQuadraticForm.model_validate(
            {
                **_form(),
                "cross_terms": [
                    {"left": 0, "right": 1, "coefficient": _rational(0)},
                ],
            }
        )
    with pytest.raises(ValidationError, match="ordered"):
        RationalQuadraticForm.model_validate(
            {
                "axis": ["x", "y", "z"],
                "diagonal_coefficients": [_rational(0), _rational(0), _rational(0)],
                "cross_terms": [
                    {"left": 1, "right": 2, "coefficient": _rational(1)},
                    {"left": 0, "right": 1, "coefficient": _rational(1)},
                ],
            }
        )


def test_result_replays_the_source_bound_value() -> None:
    with pytest.raises(ValidationError, match="exact quadratic-form evaluation"):
        EvaluationResult.model_validate(
            {"form": _form(), "vector": _vector(), "value": _rational(1)}
        )


def test_digit_bounds_reject_before_exact_arithmetic() -> None:
    too_large_coefficient = "1" + "0" * MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS
    with pytest.raises(ValidationError, match="cross coefficient"):
        RationalQuadraticForm.model_validate(
            {
                "axis": ["x", "y"],
                "diagonal_coefficients": [_rational(0), _rational(0)],
                "cross_terms": [
                    {
                        "left": 0,
                        "right": 1,
                        "coefficient": {"num": too_large_coefficient, "den": "1"},
                    }
                ],
            }
        )
    too_large_coordinate = "1" + "0" * MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS
    with pytest.raises(ValidationError, match="vector coordinate"):
        RationalCoordinateVector.model_validate(
            {
                "axis": ["x"],
                "coordinates": [{"num": too_large_coordinate, "den": "1"}],
            }
        )


def test_evaluation_preflights_the_aggregate_denominator() -> None:
    denominator = "1" + "0" * (MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS - 1)
    dimension = (
        MAX_QUADRATIC_EVALUATION_COMMON_DENOMINATOR_DIGITS
        // MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS
        + 1
    )
    assert dimension <= 32
    labels = [f"x{index}" for index in range(dimension)]
    form = {
        "axis": labels,
        "diagonal_coefficients": [{"num": "1", "den": denominator} for _ in labels],
    }
    vector = {
        "axis": labels,
        "coordinates": [_rational(1) for _ in labels],
    }

    with pytest.raises(ValidationError, match="aggregate denominator"):
        EvaluationRequest.model_validate({"form": form, "vector": vector})
