"""Exact behavior of coordinate-subspace restrictions."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.number_theory.quadratic_forms.general._models import (
    EvaluationRequest,
)
from jacobian.math.number_theory.quadratic_forms.general._tools import evaluate_form
from jacobian.math.number_theory.quadratic_forms.general.direct_sum_models import (
    QuadraticFormRestrictionRequest,
    QuadraticFormRestrictionResult,
)
from jacobian.math.number_theory.quadratic_forms.general.direct_sum_operations import (
    quadratic_form_restrict_coordinates,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalCoordinateVector,
    RationalQuadraticForm,
)


def _r(value: int) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(value, 1)


def _form() -> RationalQuadraticForm:
    return RationalQuadraticForm(
        axis=("x", "y", "z"),
        diagonal_coefficients=(_r(2), _r(5), _r(7)),
        cross_terms=(
            QuadraticCrossTerm(left=0, right=1, coefficient=_r(3)),
            QuadraticCrossTerm(left=0, right=2, coefficient=_r(-4)),
            QuadraticCrossTerm(left=1, right=2, coefficient=_r(6)),
        ),
    )


def test_restriction_preserves_selected_order_and_exact_inclusion() -> None:
    source = _form()
    result = quadratic_form_restrict_coordinates(
        QuadraticFormRestrictionRequest(form=source, selected_axis=("z", "x"))
    )
    assert result.source_form == source
    assert result.form.axis == ("z", "x")
    assert result.form.diagonal_coefficients == (_r(7), _r(2))
    assert result.form.cross_terms == (
        QuadraticCrossTerm(left=0, right=1, coefficient=_r(-4)),
    )
    assert result.inclusion.entries == (
        (_r(0), _r(1)),
        (_r(0), _r(0)),
        (_r(1), _r(0)),
    )

    target_vector = RationalCoordinateVector(
        axis=("z", "x"), coordinates=(_r(2), _r(3))
    )
    source_vector = RationalCoordinateVector(
        axis=source.axis, coordinates=(_r(3), _r(0), _r(2))
    )
    target_value = evaluate_form(
        EvaluationRequest(form=result.form, vector=target_vector)
    ).value
    source_value = evaluate_form(
        EvaluationRequest(form=source, vector=source_vector)
    ).value
    assert target_value == source_value
    assert (
        QuadraticFormRestrictionResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_restriction_to_empty_subset_is_zero_dimensional_zero_form() -> None:
    result = quadratic_form_restrict_coordinates(
        QuadraticFormRestrictionRequest(form=_form(), selected_axis=())
    )
    assert result.form.axis == ()
    assert result.form.diagonal_coefficients == ()
    assert result.form.cross_terms == ()
    assert result.inclusion.row_count == 3
    assert result.inclusion.column_count == 0
    assert result.inclusion.entries == ((), (), ())


def test_empty_form_restricts_to_empty_form() -> None:
    form = RationalQuadraticForm(axis=(), diagonal_coefficients=())
    result = quadratic_form_restrict_coordinates(
        QuadraticFormRestrictionRequest(form=form, selected_axis=())
    )
    assert result.form == form
    assert result.inclusion.row_count == result.inclusion.column_count == 0
    assert result.inclusion.entries == ()


@pytest.mark.parametrize(
    ("selected_axis", "message"),
    [(("x", "x"), "must be unique"), (("unknown",), "belong to the source axis")],
)
def test_request_rejects_repeated_or_foreign_labels(selected_axis, message) -> None:
    with pytest.raises(ValueError, match=message):
        QuadraticFormRestrictionRequest(form=_form(), selected_axis=selected_axis)


def test_deserialization_rejects_inclusion_that_does_not_select_declared_axes() -> None:
    result = quadratic_form_restrict_coordinates(
        QuadraticFormRestrictionRequest(form=_form(), selected_axis=("z", "x"))
    )
    payload = result.model_dump(mode="python")
    payload["inclusion"]["entries"] = (
        (_r(1), _r(0)),
        (_r(0), _r(0)),
        (_r(0), _r(1)),
    )
    with pytest.raises(ValueError, match="must select the declared source axes"):
        QuadraticFormRestrictionResult.model_validate(payload)
