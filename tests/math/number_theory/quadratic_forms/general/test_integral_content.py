from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.number_theory.quadratic_forms.general._models import (
    IntegralContentRequest,
)
from jacobian.math.number_theory.quadratic_forms.general.operations import (
    integral_coefficient_content,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalQuadraticForm,
)


def _q(value: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(value, denominator)


def test_integral_content_and_primitive_part_include_cross_coefficients() -> None:
    form = RationalQuadraticForm(
        axis=("x", "y", "z"),
        diagonal_coefficients=(_q(-12), _q(0), _q(18)),
        cross_terms=(QuadraticCrossTerm(left=0, right=2, coefficient=_q(30)),),
    )

    result = integral_coefficient_content(form)

    assert result.content == 6
    assert tuple(
        value.as_fraction() for value in result.primitive_part.diagonal_coefficients
    ) == (Fraction(-2), Fraction(0), Fraction(3))
    assert result.primitive_part.cross_terms[0].coefficient.as_fraction() == 5
    assert result.primitive_part.axis == form.axis


def test_large_content_uses_decimal_string_json_encoding() -> None:
    form = RationalQuadraticForm(axis=("x",), diagonal_coefficients=(_q(10**20),))
    result = integral_coefficient_content(form)
    assert result.content == 10**20
    assert result.model_dump(mode="json")["content"] == "100000000000000000000"


def test_zero_form_has_zero_content_and_zero_primitive_part() -> None:
    form = RationalQuadraticForm(axis=("x",), diagonal_coefficients=(_q(0),))
    result = integral_coefficient_content(form)
    assert result.content == 0
    assert result.primitive_part == form


def test_integral_content_rejects_rational_coefficients_and_oversized_support() -> None:
    rational = RationalQuadraticForm(axis=("x",), diagonal_coefficients=(_q(1, 2),))
    with pytest.raises(ValueError, match="integer polynomial coefficients"):
        IntegralContentRequest(form=rational)

    axis = tuple(f"x{i}" for i in range(4097))
    oversized = RationalQuadraticForm(
        axis=axis, diagonal_coefficients=tuple(_q(1) for _ in axis)
    )
    with pytest.raises(ValueError, match="support exceeds 4096"):
        IntegralContentRequest(form=oversized)
