"""Exact and contract tests for rational quadratic-form evaluation."""

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.quadratic_forms import (
    QuadraticCrossTerm,
    RationalCoordinateVector,
    RationalQuadraticForm,
    evaluate_rational_quadratic_form,
)
from jacobian.math.quadratic_forms._models import EvaluationRequest, EvaluationResult
from jacobian.math.quadratic_forms._operations import (
    _verify_evaluation_result,
    evaluate_form,
)
from jacobian.math.quadratic_forms._tools import TOOLS
from jacobian.math.quadratic_forms.values import (
    MAX_QUADRATIC_EVALUATION_DIGITS,
    MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS,
    MAX_QUADRATIC_EVALUATION_TERM_DIGITS,
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


def test_zero_dimensional_form_evaluates_to_zero() -> None:
    request = EvaluationRequest.model_validate(
        {
            "form": {"axis": [], "diagonal_coefficients": []},
            "vector": {"axis": [], "coordinates": []},
        }
    )

    assert evaluate_rational_quadratic_form(request.form, request.vector) == Fraction(0)
    assert evaluate_form(request).value == CanonicalRational(num="0", den="1")


def test_zero_dimensional_axis_keeps_coupled_length_validation() -> None:
    with pytest.raises(ValidationError) as error:
        RationalQuadraticForm.model_validate(
            {"axis": [], "diagonal_coefficients": [_rational(1)]}
        )
    assert error.value.errors()[0]["type"] == "quadratic_form.diagonal_length_mismatch"
    with pytest.raises(ValidationError) as error:
        RationalCoordinateVector.model_validate({"axis": ["u"], "coordinates": []})
    assert (
        error.value.errors()[0]["type"] == "quadratic_form.coordinate_length_mismatch"
    )


def test_axis_mismatch_is_rejected_before_arithmetic() -> None:
    with pytest.raises(ValidationError) as error:
        EvaluationRequest.model_validate(
            {
                "form": _form(),
                "vector": {
                    "axis": ["y", "x"],
                    "coordinates": [_rational(2), _rational(1)],
                },
            }
        )
    assert error.value.errors()[0]["type"] == "quadratic_form.axis_mismatch"


def test_cross_terms_are_nonzero_unique_and_canonically_ordered() -> None:
    with pytest.raises(ValidationError) as error:
        RationalQuadraticForm.model_validate(
            {
                **_form(),
                "cross_terms": [
                    {"left": 0, "right": 1, "coefficient": _rational(0)},
                ],
            }
        )
    assert error.value.errors()[0]["type"] == "quadratic_form.zero_cross_term"
    with pytest.raises(ValidationError) as error:
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
    assert error.value.errors()[0]["type"] == "quadratic_form.cross_terms_not_canonical"
    with pytest.raises(ValidationError) as error:
        RationalQuadraticForm.model_validate(
            {
                "axis": ["x"],
                "diagonal_coefficients": [_rational(1)],
                "cross_terms": [
                    {"left": 0, "right": 3, "coefficient": _rational(1)},
                ],
            }
        )
    assert error.value.errors()[0]["type"] == "quadratic_form.cross_term_out_of_range"


def test_result_parsing_keeps_the_bounded_structural_contract() -> None:
    assert EvaluationResult.model_validate(
        {"form": _form(), "vector": _vector(), "value": _rational(1)}
    ).value.as_fraction() == Fraction(1)


def test_digit_bounds_reject_before_exact_arithmetic() -> None:
    too_large_coefficient = "1" + "0" * MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS
    with pytest.raises(ValidationError) as error:
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
    assert error.value.errors()[0]["type"] == "quadratic_form.coefficient_bound"
    too_large_coordinate = "1" + "0" * MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS
    with pytest.raises(ValidationError) as error:
        RationalCoordinateVector.model_validate(
            {
                "axis": ["x"],
                "coordinates": [{"num": too_large_coordinate, "den": "1"}],
            }
        )
    assert error.value.errors()[0]["type"] == "quadratic_form.coordinate_bound"


def test_work_based_admission_accepts_light_high_dimension_forms() -> None:
    labels = [f"x{index}" for index in range(33)]
    form = {
        "axis": labels,
        "diagonal_coefficients": [_rational(1) for _ in labels],
    }
    vector = {
        "axis": labels,
        "coordinates": [_rational(index + 1) for index in range(33)],
    }
    request = EvaluationRequest.model_validate({"form": form, "vector": vector})

    assert evaluate_rational_quadratic_form(request.form, request.vector) == Fraction(
        sum((index + 1) ** 2 for index in range(33))
    )


def test_evaluation_budget_admits_forms_near_the_denominator_boundary() -> None:
    denominator = "1" + "0" * (MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS - 1)
    densest_terms = (
        MAX_QUADRATIC_EVALUATION_DIGITS - MAX_QUADRATIC_EVALUATION_TERM_DIGITS
    ) // MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS
    axes = (
        MAX_QUADRATIC_EVALUATION_DIGITS
        - MAX_QUADRATIC_EVALUATION_TERM_DIGITS
        - len(str(densest_terms))
    ) // MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS
    labels = [f"x{index}" for index in range(axes)]
    form = {
        "axis": labels,
        "diagonal_coefficients": [{"num": "1", "den": denominator} for _ in labels],
    }
    vector = {
        "axis": labels,
        "coordinates": [_rational(1) for _ in labels],
    }
    request = EvaluationRequest.model_validate({"form": form, "vector": vector})

    assert evaluate_rational_quadratic_form(request.form, request.vector) == Fraction(
        axes, 10 ** (MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS - 1)
    )


def test_evaluation_preflights_the_aggregate_denominator() -> None:
    denominator = "1" + "0" * (MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS - 1)
    dimension = (
        MAX_QUADRATIC_EVALUATION_DIGITS - MAX_QUADRATIC_EVALUATION_TERM_DIGITS
    ) // MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS + 1
    labels = [f"x{index}" for index in range(dimension)]
    form = {
        "axis": labels,
        "diagonal_coefficients": [{"num": "1", "den": denominator} for _ in labels],
    }
    vector = {
        "axis": labels,
        "coordinates": [_rational(1) for _ in labels],
    }

    with pytest.raises(ValidationError) as error:
        EvaluationRequest.model_validate({"form": form, "vector": vector})
    assert error.value.errors()[0]["type"] == "quadratic_form.evaluation_budget"


def test_evaluation_budget_ignores_annihilated_monomials() -> None:
    denominator = "1" + "0" * (MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS - 1)
    labels = [f"x{index}" for index in range(30)]
    form = {
        "axis": labels,
        "diagonal_coefficients": [{"num": "1", "den": denominator} for _ in labels],
        "cross_terms": [
            {"left": 0, "right": 1, "coefficient": {"num": "1", "den": denominator}},
        ],
    }
    vector = {
        "axis": labels,
        "coordinates": [_rational(0) for _ in labels],
    }

    assert evaluate_form(
        EvaluationRequest.model_validate({"form": form, "vector": vector})
    ).value == CanonicalRational(num="0", den="1")


def test_zero_coordinates_exclude_irrelevant_coefficient_denominators() -> None:
    active_denominator = "1" + "0" * (MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS // 2 - 1)
    annihilated_denominator = "1" + "0" * (MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS - 1)
    labels = [f"a{index}" for index in range(20)] + [f"b{index}" for index in range(20)]
    form = {
        "axis": labels,
        "diagonal_coefficients": [
            {"num": "1", "den": active_denominator} for _ in range(20)
        ]
        + [{"num": "1", "den": annihilated_denominator} for _ in range(20)],
    }
    vector = {
        "axis": labels,
        "coordinates": [_rational(1) for _ in range(20)]
        + [_rational(0) for _ in range(20)],
    }
    request = EvaluationRequest.model_validate({"form": form, "vector": vector})

    assert evaluate_rational_quadratic_form(request.form, request.vector) == Fraction(
        20, 10 ** (MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS // 2 - 1)
    )


def test_form_schema_documents_coupled_axis_invariants() -> None:
    properties = RationalQuadraticForm.model_json_schema()["properties"]

    assert "labels must be unique" in properties["axis"]["description"]
    assert (
        "exactly one coefficient per axis label"
        in properties["diagonal_coefficients"]["description"]
    )
    assert (
        "every cross-term index must lie within the declared axis"
        in properties["cross_terms"]["description"]
    )


def test_schema_documents_entry_digit_limits() -> None:
    form_properties = RationalQuadraticForm.model_json_schema()["properties"]
    vector_properties = RationalCoordinateVector.model_json_schema()["properties"]

    assert (
        f"at most {MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS} decimal digits"
        in form_properties["diagonal_coefficients"]["description"]
    )
    assert (
        f"at most {MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS} decimal digits"
        in QuadraticCrossTerm.model_json_schema()["properties"]["coefficient"][
            "description"
        ]
    )
    assert (
        f"at most {MAX_QUADRATIC_VECTOR_COORDINATE_DIGITS} decimal digits"
        in vector_properties["coordinates"]["description"]
    )


def test_total_support_bound_rejects_unbounded_annihilated_support() -> None:
    dimension = MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS + 1
    labels = [f"x{index}" for index in range(dimension)]
    form = {
        "axis": labels,
        "diagonal_coefficients": [_rational(0) for _ in labels],
    }
    vector = {
        "axis": labels,
        "coordinates": [_rational(0) for _ in labels],
    }

    with pytest.raises(ValidationError) as error:
        EvaluationRequest.model_validate({"form": form, "vector": vector})
    assert error.value.errors()[0]["type"] == "quadratic_form.support_budget"
    result = EvaluationResult.model_validate(
        {"form": form, "vector": vector, "value": _rational(0)}
    )
    assert not _verify_evaluation_result(result)


def test_total_support_admits_the_boundary_and_rejects_one_past_it() -> None:
    def _request(dimension: int) -> dict[str, object]:
        labels = [f"x{index}" for index in range(dimension)]
        return {
            "form": {
                "axis": labels,
                "diagonal_coefficients": [
                    _rational(i + 1) for i, _ in enumerate(labels)
                ],
            },
            "vector": {
                "axis": labels,
                "coordinates": [_rational(0) for _ in labels],
            },
        }

    accepted = EvaluationRequest.model_validate(
        _request(MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS)
    )

    assert evaluate_rational_quadratic_form(accepted.form, accepted.vector) == Fraction(
        0
    )
    with pytest.raises(ValidationError) as error:
        EvaluationRequest.model_validate(
            _request(MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS + 1)
        )
    assert error.value.errors()[0]["type"] == "quadratic_form.support_budget"


def test_request_schema_documents_the_evaluation_budgets() -> None:
    properties = EvaluationRequest.model_json_schema()["properties"]

    assert (
        f"at {MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS} terms"
        in properties["form"]["description"]
    )
    assert (
        f"d + {MAX_QUADRATIC_EVALUATION_TERM_DIGITS} + len(str(t)) <= "
        f"{MAX_QUADRATIC_EVALUATION_DIGITS}" in properties["vector"]["description"]
    )


def test_tool_description_and_example_publish_the_evaluation_budgets() -> None:
    tool = TOOLS[0]

    assert f"{MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS} terms" in tool.description
    assert (
        f"d + {MAX_QUADRATIC_EVALUATION_TERM_DIGITS} + len(str(t)) <= "
        f"{MAX_QUADRATIC_EVALUATION_DIGITS}" in tool.description
    )
    example_description = tool.examples[0].description
    assert f"{MAX_QUADRATIC_EVALUATION_SUPPORT_TERMS} terms" in example_description
    assert (
        f"d + {MAX_QUADRATIC_EVALUATION_TERM_DIGITS} + digits(t) within "
        f"{MAX_QUADRATIC_EVALUATION_DIGITS}" in example_description
    )


def test_vector_schema_documents_coupled_axis_invariants() -> None:
    properties = RationalCoordinateVector.model_json_schema()["properties"]

    assert "labels must be unique" in properties["axis"]["description"]
    assert (
        "exactly one coordinate per axis label"
        in properties["coordinates"]["description"]
    )
