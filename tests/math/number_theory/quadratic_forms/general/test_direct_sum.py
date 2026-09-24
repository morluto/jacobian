"""Real exact behavior for direct sums of rational quadratic forms."""

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.number_theory.quadratic_forms.general._models import (
    EvaluationRequest,
)
from jacobian.math.number_theory.quadratic_forms.general._tools import evaluate_form
from jacobian.math.number_theory.quadratic_forms.general.direct_sum_models import (
    MAX_DIRECT_SUM_AXIS,
    MAX_DIRECT_SUM_FORM_TERMS,
    MAX_DIRECT_SUM_OUTPUT_DIGITS,
    QuadraticFormDirectSumRequest,
    QuadraticFormDirectSumResult,
    direct_sum_output_digit_upper_bound,
)
from jacobian.math.number_theory.quadratic_forms.general.direct_sum_operations import (
    quadratic_form_direct_sum,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalCoordinateVector,
    RationalQuadraticForm,
)


def _r(value: int) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(value, 1)


def _form(
    axis: tuple[str, ...], diagonal: tuple[int, ...], cross=()
) -> RationalQuadraticForm:
    return RationalQuadraticForm(
        axis=axis,
        diagonal_coefficients=tuple(_r(value) for value in diagonal),
        cross_terms=tuple(
            QuadraticCrossTerm(left=i, right=j, coefficient=_r(value))
            for i, j, value in cross
        ),
    )


def test_direct_sum_preserves_factor_order_and_separates_reused_labels() -> None:
    first = _form(("x", "y"), (1, 2), ((0, 1, 3),))
    second = _form(("x",), (5,))
    result = quadratic_form_direct_sum(
        QuadraticFormDirectSumRequest(forms=(first, second))
    )

    assert result.source_forms == (first, second)
    assert result.form.axis == ("qf0_0", "qf0_1", "qf1_0")
    assert result.form.diagonal_coefficients == (_r(1), _r(2), _r(5))
    assert result.form.cross_terms == (
        QuadraticCrossTerm(left=0, right=1, coefficient=_r(3)),
    )
    assert result.coordinate_inclusions[0].entries == (
        (_r(1), _r(0)),
        (_r(0), _r(1)),
        (_r(0), _r(0)),
    )
    assert result.coordinate_projections[1].entries == ((_r(0), _r(0), _r(1)),)

    coordinates = (_r(2), _r(-1), _r(3))
    direct = evaluate_form(
        EvaluationRequest(
            form=result.form,
            vector=RationalCoordinateVector(
                axis=result.form.axis, coordinates=coordinates
            ),
        )
    )
    first_value = evaluate_form(
        EvaluationRequest(
            form=first,
            vector=RationalCoordinateVector(
                axis=first.axis, coordinates=coordinates[:2]
            ),
        )
    )
    second_value = evaluate_form(
        EvaluationRequest(
            form=second,
            vector=RationalCoordinateVector(
                axis=second.axis, coordinates=coordinates[2:]
            ),
        )
    )
    assert direct.value.as_fraction() == (
        first_value.value.as_fraction() + second_value.value.as_fraction()
    )
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_empty_sum_is_the_zero_dimensional_form() -> None:
    result = quadratic_form_direct_sum(QuadraticFormDirectSumRequest(forms=()))
    assert result.form.axis == ()
    assert result.form.diagonal_coefficients == ()
    assert result.coordinate_inclusions == result.coordinate_projections == ()


def test_zero_dimensional_summand_retains_zero_by_nonzero_projection_shape() -> None:
    scalar = _form(("t",), (7,))
    result = quadratic_form_direct_sum(
        QuadraticFormDirectSumRequest(forms=(_form((), ()), scalar))
    )
    assert result.coordinate_inclusions[0].row_count == 1
    assert result.coordinate_inclusions[0].column_count == 0
    assert result.coordinate_projections[0].row_count == 0
    assert result.coordinate_projections[0].column_count == 1
    assert result.form.diagonal_coefficients == (_r(7),)


def test_direct_sum_admits_aggregate_axis_before_building_maps() -> None:
    oversized = _form(tuple(f"x{i}" for i in range(129)), (0,) * 129)
    with pytest.raises(ValueError, match="aggregate axis bound"):
        QuadraticFormDirectSumRequest(forms=(oversized,))


def test_inclusion_projection_composition_is_identity_on_each_factor() -> None:
    forms = (_form(("a", "b"), (0, 0)), _form(("c",), (0,)))
    result = quadratic_form_direct_sum(QuadraticFormDirectSumRequest(forms=forms))
    for inclusion, projection, source in zip(
        result.coordinate_inclusions,
        result.coordinate_projections,
        forms,
        strict=True,
    ):
        product = tuple(
            tuple(
                sum(
                    projection.entries[i][k].as_fraction()
                    * inclusion.entries[k][j].as_fraction()
                    for k in range(len(result.form.axis))
                )
                for j in range(len(source.axis))
            )
            for i in range(len(source.axis))
        )
        assert product == tuple(
            tuple(Fraction(int(i == j)) for j in range(len(source.axis)))
            for i in range(len(source.axis))
        )


def test_deserialization_rejects_inconsistent_coordinate_map_shape() -> None:
    result = quadratic_form_direct_sum(
        QuadraticFormDirectSumRequest(forms=(_form(("x",), (1,)),))
    )
    payload = result.model_dump(mode="python")
    payload["coordinate_inclusions"][0]["column_count"] = 0
    payload["coordinate_inclusions"][0]["entries"] = ((),)
    with pytest.raises(ValueError, match="inclusion has inconsistent dimensions"):
        QuadraticFormDirectSumResult.model_validate(payload)


def test_maximum_admitted_support_and_dense_maps_fit_output_digit_limit() -> None:
    assert (
        direct_sum_output_digit_upper_bound(
            MAX_DIRECT_SUM_AXIS, 2 * MAX_DIRECT_SUM_FORM_TERMS
        )
        < MAX_DIRECT_SUM_OUTPUT_DIGITS
    )
    assert (
        direct_sum_output_digit_upper_bound(
            MAX_DIRECT_SUM_AXIS,
            2 * MAX_DIRECT_SUM_FORM_TERMS,
            map_component_digits=32_768,
        )
        > MAX_DIRECT_SUM_OUTPUT_DIGITS
    )


def test_deserialization_rejects_maps_beyond_the_output_digit_limit() -> None:
    axis = tuple(f"x{i}" for i in range(16))
    result = quadratic_form_direct_sum(
        QuadraticFormDirectSumRequest(forms=(_form(axis, (1,) * 16),))
    )
    payload = result.model_dump(mode="python")
    payload["coordinate_inclusions"][0]["entries"] = tuple(
        tuple(CanonicalRational.from_integer_ratio(10**9_000, 1) for _ in range(16))
        for _ in range(16)
    )
    with pytest.raises(ValueError, match="output digit bound"):
        QuadraticFormDirectSumResult.model_validate(payload)
