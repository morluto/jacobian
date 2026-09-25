from __future__ import annotations

from fractions import Fraction
from itertools import combinations, islice

import pytest

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    ThetaSeriesPrefixRequest,
)
from jacobian.math.number_theory.quadratic_forms.general._tools import (
    TOOLS,
    compute_scale,
)
from jacobian.math.number_theory.quadratic_forms.general.scaling_models import (
    MAX_QUADRATIC_SCALE_OUTPUT_BYTES,
    QuadraticFormScaleRequest,
    QuadraticFormScaleResult,
)
from jacobian.math.number_theory.quadratic_forms.general.scaling_operations import (
    scale_rational_quadratic_form,
)
from jacobian.math.number_theory.quadratic_forms.general.theta_operations import (
    theta_series_prefix,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
    QuadraticCrossTerm,
    RationalQuadraticForm,
)


def _q(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(numerator, denominator)


def _form(
    diagonal: tuple[CanonicalRational, ...],
    cross: tuple[tuple[int, int, CanonicalRational], ...] = (),
) -> RationalQuadraticForm:
    return RationalQuadraticForm(
        axis=tuple("xyz"[: len(diagonal)]),
        diagonal_coefficients=diagonal,
        cross_terms=tuple(
            QuadraticCrossTerm(left=i, right=j, coefficient=value)
            for i, j, value in cross
        ),
    )


def test_scale_matches_independent_polynomial_and_evaluation_oracles() -> None:
    source = _form(
        (_q(3, 2), _q(4, 5)),
        ((0, 1, _q(1, 3)),),
    )
    request = QuadraticFormScaleRequest(form=source, factor=_q(6, 5))
    result = scale_rational_quadratic_form(request)

    assert tuple(
        value.as_fraction() for value in result.form.diagonal_coefficients
    ) == (
        Fraction(9, 5),
        Fraction(24, 25),
    )
    assert result.form.cross_terms[0].coefficient.as_fraction() == Fraction(2, 5)
    assert result.source_form == source
    assert result.form.axis == source.axis

    # The half-polar Gram matrices have determinant scaled by c^2.
    source_gram_determinant = Fraction(3, 2) * Fraction(4, 5) - Fraction(1, 6) ** 2
    scaled_gram_determinant = Fraction(9, 5) * Fraction(24, 25) - Fraction(1, 5) ** 2
    assert source_gram_determinant == Fraction(211, 180)
    assert scaled_gram_determinant == Fraction(211, 125)
    assert scaled_gram_determinant == Fraction(6, 5) ** 2 * source_gram_determinant

    # Evaluate the defining relation independently over a small rational grid.
    for x in (Fraction(-2, 3), Fraction(0), Fraction(5, 4)):
        for y in (Fraction(-1, 2), Fraction(1, 3)):
            old = (
                Fraction(3, 2) * x * x + Fraction(1, 3) * x * y + Fraction(4, 5) * y * y
            )
            scaled = (
                result.form.diagonal_coefficients[0].as_fraction() * x * x
                + result.form.cross_terms[0].coefficient.as_fraction() * x * y
                + result.form.diagonal_coefficients[1].as_fraction() * y * y
            )
            assert scaled == Fraction(6, 5) * old


def test_scaled_integral_form_composes_through_theta_after_json_roundtrip() -> None:
    source = _form((_q(1), _q(1)))
    result = compute_scale(QuadraticFormScaleRequest(form=source, factor=_q(2)))
    decoded = QuadraticFormScaleResult.model_validate_json(result.model_dump_json())
    theta = theta_series_prefix(ThetaSeriesPrefixRequest(form=decoded.form, cutoff=6))

    # Independent finite enumeration in [-2,2]^2 contains every vector with
    # 2*(x^2+y^2) <= 6, since either coordinate of magnitude 2 already exceeds 6.
    expected = [0] * 7
    for x in range(-2, 3):
        for y in range(-2, 3):
            value = 2 * (x * x + y * y)
            if value <= 6:
                expected[value] += 1
    assert theta.coefficients == tuple(expected) == (1, 0, 4, 0, 4, 0, 0)


def test_zero_scale_omits_cross_terms_and_keeps_axis_and_diagonal_shape() -> None:
    source = _form((_q(3), _q(-2)), ((0, 1, _q(7)),))
    result = scale_rational_quadratic_form(
        QuadraticFormScaleRequest(form=source, factor=_q(0))
    )
    assert result.form.axis == source.axis
    assert result.form.diagonal_coefficients == (_q(0), _q(0))
    assert result.form.cross_terms == ()


def test_scale_preflight_uses_exact_cancellation_and_rejects_true_growth() -> None:
    large = 10 ** (MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS - 1)
    canceling = _form((_q(large),))
    result = scale_rational_quadratic_form(
        QuadraticFormScaleRequest(form=canceling, factor=_q(1, large))
    )
    assert result.form.diagonal_coefficients == (_q(1),)

    growing = _form((_q(large),))
    with pytest.raises(OperationResourceAdmissionError) as error:
        scale_rational_quadratic_form(
            QuadraticFormScaleRequest(form=growing, factor=_q(10))
        )
    assert error.value.errors()[0]["type"] == (
        "quadratic_form.scale_coefficient_growth"
    )


def test_scale_output_bound_admits_ceiling_and_rejects_next_support_size() -> None:
    axis = tuple("😀" * 60 + f"{index:03d}" for index in range(128))
    coefficient = _q(10 ** (MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS - 1) - 1)
    crosses = tuple(
        QuadraticCrossTerm(left=left, right=right, coefficient=coefficient)
        for left, right in islice(combinations(range(128), 2), 947)
    )
    form = RationalQuadraticForm(
        axis=axis,
        diagonal_coefficients=(coefficient,) * 128,
        cross_terms=crosses,
    )
    result = scale_rational_quadratic_form(
        QuadraticFormScaleRequest(form=form, factor=_q(1))
    )
    assert len(result.model_dump_json().encode("utf-8")) <= (
        MAX_QUADRATIC_SCALE_OUTPUT_BYTES
    )

    over_limit = RationalQuadraticForm(
        axis=axis,
        diagonal_coefficients=(coefficient,) * 128,
        cross_terms=(
            *crosses,
            *(
                QuadraticCrossTerm(
                    left=left, right=right, coefficient=coefficient
                )
                for left, right in islice(combinations(range(128), 2), 947, 4_096)
            ),
        ),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        scale_rational_quadratic_form(
            QuadraticFormScaleRequest(form=over_limit, factor=_q(1))
        )
    assert error.value.errors()[0]["type"] == "quadratic_form.scale_support_bound"


def test_scaling_operation_is_published_in_owner_manifest() -> None:
    assert "quadratic_form.scale.compute" in {tool.operation_id for tool in TOOLS}


def test_scaling_composes_through_public_catalog_dispatch() -> None:
    catalog = Catalog.open()
    operation_id = "quadratic_form.scale.compute"
    operation = catalog.operation(operation_id)
    assert operation is not None
    request = {
        "form": {
            "axis": ["x", "y"],
            "diagonal_coefficients": [
                {"num": "1", "den": "1"},
                {"num": "1", "den": "1"},
            ],
        },
        "factor": {"num": "2", "den": "1"},
    }
    operation.request_type.model_validate_json(encode_strict_json(request), strict=True)
    response = invoke_operation(operation_id, request, catalog)
    decoded = operation.result_type.model_validate_json(
        encode_strict_json(response.output), strict=True
    )
    theta = theta_series_prefix(ThetaSeriesPrefixRequest(form=decoded.form, cutoff=4))
    assert theta.coefficients == (1, 0, 4, 0, 4)
