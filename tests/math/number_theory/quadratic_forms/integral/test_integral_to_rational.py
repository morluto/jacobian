"""Exact ZZ-to-QQ quadratic-form transport and target interoperability."""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import product
from math import gcd

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    FiniteBoxProfileRequest,
)
from jacobian.math.number_theory.quadratic_forms.general.finite_box_operations import (
    finite_box_value_profile,
)
from jacobian.math.number_theory.quadratic_forms.general.operations import (
    evaluate_rational_quadratic_form,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    RationalCoordinateVector,
)
from jacobian.math.number_theory.quadratic_forms.integral import (
    IntegralQuadraticForm,
    IntegralQuadraticFormInclusionRequest,
    integral_form_to_rational,
)


def _rational_vector(axis: tuple[str, ...], values: tuple[int, ...]):
    return RationalCoordinateVector(
        axis=axis,
        coordinates=tuple(
            CanonicalRational.from_integer_ratio(value, 1) for value in values
        ),
    )


def test_zz_inclusion_round_trips_and_composes_with_exact_box_profile() -> None:
    source = IntegralQuadraticForm(
        axis=("u", "v"),
        diagonal_coefficients=(2, 0),
        cross_terms=({"left": 0, "right": 1, "coefficient": 3},),
    )
    inclusion = integral_form_to_rational(
        IntegralQuadraticFormInclusionRequest(form=source)
    )

    restored = type(inclusion).model_validate_json(inclusion.model_dump_json())
    assert restored == inclusion
    assert restored.map == "ZZ_TO_QQ_COEFFICIENT_INCLUSION"
    assert restored.source.axis == restored.target.axis == ("u", "v")
    assert tuple(
        value.as_fraction() for value in restored.target.diagonal_coefficients
    ) == (
        Fraction(2),
        Fraction(0),
    )
    assert restored.target.cross_terms[0].coefficient.as_fraction() == 3
    assert gcd(2, gcd(0, 3)) == gcd(
        restored.target.diagonal_coefficients[0].num,
        gcd(
            restored.target.diagonal_coefficients[1].num,
            restored.target.cross_terms[0].coefficient.num,
        ),
    )

    # Independent direct polynomial oracle and the existing QQ evaluator agree
    # for all points in this exact finite box.
    for vector in product(range(-2, 3), repeat=2):
        expected = 2 * vector[0] ** 2 + 3 * vector[0] * vector[1]
        actual = evaluate_rational_quadratic_form(
            restored.target, _rational_vector(restored.target.axis, vector)
        )
        assert actual == Fraction(expected)

    profile = finite_box_value_profile(
        FiniteBoxProfileRequest(form=restored.target, radius=1)
    )
    actual_profile = {row.value: row.representation_count for row in profile.rows}
    expected_profile: dict[int, int] = {}
    for vector in product(range(-1, 2), repeat=2):
        value = 2 * vector[0] ** 2 + 3 * vector[0] * vector[1]
        expected_profile[value] = expected_profile.get(value, 0) + 1
    assert actual_profile == expected_profile


@pytest.mark.parametrize(
    "axis,diagonal,cross_terms",
    [
        ((), (), ()),
        (("x", "y"), (0, 0), ()),
        (("x", "y"), (1, 0), ()),  # rank-one degenerate form
    ],
)
def test_empty_zero_and_degenerate_forms_keep_their_parent_and_axis(
    axis, diagonal, cross_terms
) -> None:
    form = IntegralQuadraticForm(
        axis=axis,
        diagonal_coefficients=diagonal,
        cross_terms=cross_terms,
    )
    inclusion = integral_form_to_rational(
        IntegralQuadraticFormInclusionRequest(form=form)
    )
    assert inclusion.source.domain == "ZZ"
    assert inclusion.target.domain == "QQ"
    assert inclusion.target.axis == axis
    assert (
        tuple(value.num for value in inclusion.target.diagonal_coefficients) == diagonal
    )
    assert all(value.den == 1 for value in inclusion.target.diagonal_coefficients)


def test_integral_carrier_rejects_nonintegral_wire_coefficients() -> None:
    with pytest.raises(ValueError):
        IntegralQuadraticForm.model_validate_json(
            json.dumps(
                {
                    "axis": ["x"],
                    "diagonal_coefficients": [{"num": "1", "den": "2"}],
                }
            )
        )


def test_target_output_is_admitted_before_rational_form_construction(
    monkeypatch,
) -> None:
    import jacobian.math.number_theory.quadratic_forms.integral.operations as operations

    axis = tuple(f"x{index}" for index in range(64))
    pairs = tuple((left, right) for left in range(64) for right in range(left + 1, 64))
    source = IntegralQuadraticForm(
        axis=axis,
        diagonal_coefficients=(0,) * 64,
        cross_terms=tuple(
            {"left": left, "right": right, "coefficient": 10**255}
            for left, right in pairs[: 2_048 - 64]
        ),
    )

    def no_rational_target(**kwargs):
        raise AssertionError("QQ target construction began before output admission")

    monkeypatch.setattr(operations, "RationalQuadraticForm", no_rational_target)
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        integral_form_to_rational(IntegralQuadraticFormInclusionRequest(form=source))
    assert exc_info.value.errors()[0]["type"] == (
        "quadratic_form.integral.rational_extension_output_bound"
    )


def test_catalog_operation_is_discoverable_and_its_example_runs() -> None:
    operation_id = "quadratic_form.integral.rational_extension.compute"
    tool = next(item for item in BUILTIN_TOOLS if item.operation_id == operation_id)
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert result.target.cross_terms[0].coefficient.as_fraction() == 3
