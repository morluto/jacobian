"""Exact bounded enumeration of quadratic-form values on integer boxes."""

import json
from collections import Counter
from fractions import Fraction
from itertools import product

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    FiniteBoxProfileRequest,
    FiniteBoxProfileResult,
)
from jacobian.math.number_theory.quadratic_forms.general.finite_box_operations import (
    finite_box_value_profile,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
    QuadraticCrossTerm,
    RationalQuadraticForm,
)


def _form(axis, diagonal, crosses=()):
    return RationalQuadraticForm(
        axis=axis,
        diagonal_coefficients=tuple(
            CanonicalRational.from_fraction(value) for value in diagonal
        ),
        cross_terms=tuple(
            QuadraticCrossTerm(
                left=i,
                right=j,
                coefficient=CanonicalRational.from_fraction(value),
            )
            for i, j, value in crosses
        ),
    )


def test_native_boundary_rejects_negative_radius_even_for_zero_dimensional_form():
    empty = _form((), ())
    request = FiniteBoxProfileRequest.model_construct(form=empty, radius=-1)
    with pytest.raises(OperationDomainValidationError):
        finite_box_value_profile(request)


def test_finite_box_profile_matches_independent_polar_matrix_oracle():
    form = _form(("u", "v", "w"), (2, -1, 0), ((0, 1, 3), (1, 2, -2)))
    radius = 2
    result = finite_box_value_profile(FiniteBoxProfileRequest(form=form, radius=radius))

    # Q(x)=x^T B_Q x/2, where B_Q has diagonal 2*a_i and off-diagonal
    # polynomial cross coefficients. This independent oracle uses matrix form.
    polar = ((4, 3, 0), (3, -2, -2), (0, -2, 0))
    expected = Counter()
    for vector in product(range(-radius, radius + 1), repeat=3):
        doubled = sum(
            vector[i] * polar[i][j] * vector[j] for i in range(3) for j in range(3)
        )
        assert doubled % 2 == 0
        expected[doubled // 2] += 1

    assert result.form == form
    assert result.coordinate_bounds == ((-2, 2),) * 3
    assert result.vector_count == 125
    assert {row.value: row.representation_count for row in result.rows} == expected
    assert sum(row.representation_count for row in result.rows) == 125
    assert result.minimum_value == min(expected)
    assert result.maximum_value == max(expected)


def test_zero_dimensional_and_radius_zero_profiles_are_complete():
    empty = _form((), ())
    result = finite_box_value_profile(FiniteBoxProfileRequest(form=empty, radius=0))
    assert result.vector_count == 1
    assert result.rows[0].value == 0
    assert result.rows[0].representation_count == 1

    one_axis = _form(("coordinate",), (7,))
    origin_only = finite_box_value_profile(
        FiniteBoxProfileRequest(form=one_axis, radius=0)
    )
    assert origin_only.rows[0].value == 0
    assert origin_only.coordinate_bounds == ((0, 0),)


def test_nonintegral_form_is_rejected_before_enumeration():
    form = _form(("x",), (Fraction(1, 2),))
    with pytest.raises(OperationDomainValidationError) as exc_info:
        finite_box_value_profile(FiniteBoxProfileRequest(form=form, radius=1))
    assert (
        exc_info.value.errors()[0]["type"]
        == "quadratic_form.finite_box_requires_integral"
    )


def test_box_vector_count_is_admitted_before_evaluation():
    form = _form(("x", "y", "z", "w"), (1, 1, 1, 1))
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        finite_box_value_profile(FiniteBoxProfileRequest(form=form, radius=8))
    assert (
        exc_info.value.errors()[0]["type"] == "quadratic_form.finite_box_vector_bound"
    )


def test_box_output_digit_envelope_is_admitted_before_evaluation():
    # A dense box whose value count fits, but whose tallest coefficients make
    # the aggregate profile digits exceed the output envelope before any
    # vector is evaluated.
    tall = 10**MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS - 1
    form = _form(("x", "y", "z", "w"), (tall, tall, tall, tall))
    with pytest.raises(OperationResourceAdmissionError) as exc_info:
        finite_box_value_profile(FiniteBoxProfileRequest(form=form, radius=4))
    assert (
        exc_info.value.errors()[0]["type"] == "quadratic_form.finite_box_output_bound"
    )


def test_profile_values_are_exact_decimal_integers_over_json():
    # Q(x) = 10^20 * x^2 at radius 1 returns values far outside JavaScript's
    # safe-integer range; the canonical JSON transport must carry them as
    # exact decimal strings that round-trip without rounding.
    tall = 10**20
    result = finite_box_value_profile(
        FiniteBoxProfileRequest(form=_form(("x",), (tall,)), radius=1)
    )
    payload = result.model_dump_json()
    assert '"value":"0"' in payload
    assert f'"value":"{tall}"' in payload
    assert f'"maximum_value":"{tall}"' in payload
    restored = FiniteBoxProfileResult.model_validate_json(payload)
    assert restored == result
    assert restored.rows[-1].value == tall


def test_incomplete_profile_histograms_are_rejected_on_validation():
    # The declared box [-1,1] contains three vectors, so a one-row profile
    # covering only one vector is not a complete FiniteBoxProfileResult.
    form = _form(("x",), (1,))
    base = {
        "form": json.loads(form.model_dump_json()),
        "radius": 1,
        "coordinate_bounds": [[-1, 1]],
        "vector_count": 1,
        "rows": [{"value": "0", "representation_count": 1}],
        "minimum_value": "0",
        "maximum_value": "0",
    }
    with pytest.raises(ValidationError, match="vector count must cover"):
        FiniteBoxProfileResult.model_validate_json(json.dumps(base))
    truncated = {
        **base,
        "vector_count": 3,
        "rows": [{"value": "0", "representation_count": 2}],
    }
    with pytest.raises(ValidationError, match="counts must cover"):
        FiniteBoxProfileResult.model_validate_json(json.dumps(truncated))
