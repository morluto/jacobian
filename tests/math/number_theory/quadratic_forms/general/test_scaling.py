"""Exact scalar multiplication of rational quadratic forms."""

from __future__ import annotations

from math import gcd

import pytest
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.number_theory.quadratic_forms.general.scaling_models import (
    MAX_QUADRATIC_SCALE_AXIS,
    MAX_QUADRATIC_SCALE_SUPPORT,
    QuadraticFormScaleRequest,
    QuadraticFormScaleResult,
)
from jacobian.math.number_theory.quadratic_forms.general.scaling_operations import (
    scale_rational_quadratic_form,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS,
    QuadraticCrossTerm,
    RationalQuadraticForm,
)


def _q(value: int | str, den: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(int(value), den)


def _form(
    diagonal: tuple[int | str, ...],
    cross: tuple[tuple[int, int, int | str], ...] = (),
    axis: tuple[str, ...] | None = None,
) -> RationalQuadraticForm:
    if axis is None:
        axis = tuple("xyzuvw"[: len(diagonal)]) or ("x",)
    return RationalQuadraticForm(
        axis=axis,
        diagonal_coefficients=tuple(_q(d) for d in diagonal),
        cross_terms=tuple(
            QuadraticCrossTerm(left=left, right=right, coefficient=_q(coefficient))
            for left, right, coefficient in cross
        ),
    )


def test_scaling_is_coefficientwise_over_diagonal_and_cross_terms() -> None:
    source = _form((2, 0), ((0, 1, 3),))
    result = scale_rational_quadratic_form(source, _q(3, 2))
    assert result.form.axis == source.axis
    assert result.form.diagonal_coefficients == (_q(3), _q(0))
    assert [(t.left, t.right, t.coefficient) for t in result.form.cross_terms] == [
        (0, 1, _q(9, 2))
    ]
    assert result.source_form == source
    assert result.factor == _q(3, 2)


def _evaluate(form: RationalQuadraticForm, point: dict[int, int]) -> tuple[int, int]:
    """Return Q(point) as an exact (numerator, denominator) pair."""

    total = (0, 1)
    for index, coefficient in enumerate(form.diagonal_coefficients):
        value = point.get(index, 0)
        numerator, denominator = coefficient.as_integer_ratio()
        total = _add(total, numerator * value * value, denominator)
    for term in form.cross_terms:
        numerator, denominator = term.coefficient.as_integer_ratio()
        total = _add(
            total, numerator * point[term.left] * point[term.right], denominator
        )
    return total


def _add(total: tuple[int, int], value: int, denominator: int) -> tuple[int, int]:
    """Exact QQ addition, so the oracle never leaves the rationals."""

    total_num, total_den = total
    num = total_num * denominator + value * total_den
    den = total_den * denominator
    factor = gcd(gcd(abs(num), den), abs(total_den)) or 1
    return num // factor, den // factor


def test_scaling_is_exact_over_the_rationals() -> None:
    """Independent oracle: evaluating the scaled form gives factor * Q(point).

    Checked by cross-multiplication, so no floating point is involved and the
    identity has to hold exactly for the integers involved.
    """

    source = _form((1, 2), ((0, 1, 1),))
    factor = _q(-5, 7)
    result = scale_rational_quadratic_form(source, factor)
    point = {0: 3, 1: 4}
    direct_num, direct_den = _evaluate(source, point)
    scaled_num, scaled_den = _evaluate(result.form, point)
    # factor * Q(point) == Q_scaled(point), cross-multiplied so no rounding occurs.
    factor_num, factor_den = factor.as_integer_ratio()
    assert scaled_num * direct_den * factor_den == direct_num * scaled_den * factor_num


def test_scaling_by_one_and_by_zero_are_identity_and_collapse() -> None:
    source = _form((3, 5), ((0, 1, 2),))
    assert scale_rational_quadratic_form(source, _q(1)).form == source
    zero = scale_rational_quadratic_form(source, _q(0))
    assert zero.form.diagonal_coefficients == (_q(0), _q(0))
    assert zero.form.cross_terms == ()


def test_scaling_cancels_before_multiplying() -> None:
    """A reducible factor is cancelled, so 1/2 scaled by 2/2 stays exact."""
    source = _form((1, 1), axis=("x", "y"))
    result = scale_rational_quadratic_form(source, _q(2, 2))
    assert result.form.diagonal_coefficients == (_q(1), _q(1))


def test_result_check_rejects_a_forged_coefficient() -> None:
    result = scale_rational_quadratic_form(_form((2, 0)), _q(3, 2))
    forged = result.model_copy(update={"form": _form((4, 0))})
    with pytest.raises(Exception, match="scaled diagonal"):
        forged.require_scaling_agrees_with_its_sources()


def test_result_check_accepts_the_kernel_value() -> None:
    result = scale_rational_quadratic_form(_form((2, 0)), _q(3, 2))
    assert (
        QuadraticFormScaleResult.require_scaling_agrees_with_its_sources(result)
        is result
    )


def test_axis_and_support_are_admitted_before_expansion() -> None:
    long_axis = _form(
        (1,) * (MAX_QUADRATIC_SCALE_AXIS + 1),
        axis=tuple(f"v{i}" for i in range(MAX_QUADRATIC_SCALE_AXIS + 1)),
    )
    with pytest.raises(OperationResourceAdmissionError, match="axis"):
        scale_rational_quadratic_form(long_axis, _q(1))

    # Support counts diagonal plus cross terms, so exceeding it needs many
    # distinct cross terms on an axis that is still inside the axis bound.
    width = 100
    pairs = [(i, j) for i in range(width) for j in range(i + 1, width)]
    wide = _form(
        (1,) * width,
        cross=tuple((i, j, 1) for i, j in pairs),
        axis=tuple(f"v{i}" for i in range(width)),
    )
    assert len(wide.axis) <= MAX_QUADRATIC_SCALE_AXIS
    assert len(wide.diagonal_coefficients) + len(wide.cross_terms) > (
        MAX_QUADRATIC_SCALE_SUPPORT
    )
    with pytest.raises(OperationResourceAdmissionError, match="support"):
        scale_rational_quadratic_form(wide, _q(1))


def test_an_oversized_factor_is_refused_by_its_own_width_bound() -> None:
    """A factor wider than the coefficient bound is refused as the factor."""
    huge = _q(10**MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS)
    with pytest.raises(
        OperationResourceAdmissionError, match=r"scale factor|coefficient"
    ):
        scale_rational_quadratic_form(_form((1,)), huge)


def test_coefficient_growth_is_refused_before_the_product_is_formed() -> None:
    """A wide source coefficient times a wide legal factor overflows the bound."""
    edge = 10**MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS - 1
    with pytest.raises(OperationResourceAdmissionError, match="numerator"):
        scale_rational_quadratic_form(_form((edge,)), _q(edge))


def test_scaling_by_zero_drops_cancelled_cross_terms() -> None:
    """A cross term scaled to exact zero is dropped, not retained as a zero slot.

    ``QuadraticCrossTerm`` admits only a nonzero coefficient, so the kernel
    cannot represent a cancelled term. The result contract therefore compares
    against the source positions that survive scaling rather than against every
    source position, and a genuine mismatch is still refused.
    """
    source = RationalQuadraticForm(
        axis=("x", "y"),
        diagonal_coefficients=(_q(2), _q(4)),
        cross_terms=(QuadraticCrossTerm(left=0, right=1, coefficient=_q(3)),),
    )
    zeroed = scale_rational_quadratic_form(source, _q(0))
    assert zeroed.form.cross_terms == ()
    assert zeroed.form.diagonal_coefficients == (_q(0), _q(0))
    zeroed.require_scaling_agrees_with_its_sources()

    # A scaling that keeps the cross term nonzero must still round-trip exactly.
    kept = scale_rational_quadratic_form(source, _q(2))
    kept.require_scaling_agrees_with_its_sources()
    assert kept.form.cross_terms == (
        QuadraticCrossTerm(left=0, right=1, coefficient=_q(6)),
    )
    assert kept.form.diagonal_coefficients == (_q(4), _q(8))


def test_a_forged_scaled_cross_term_position_is_still_refused() -> None:
    """Dropping cancelled terms must not weaken the surviving-term postcondition."""
    source = RationalQuadraticForm(
        axis=("x", "y"),
        diagonal_coefficients=(_q(2), _q(4)),
        cross_terms=(QuadraticCrossTerm(left=0, right=1, coefficient=_q(3)),),
    )
    forged = QuadraticFormScaleResult.model_construct(
        form=RationalQuadraticForm(
            axis=("x", "y"),
            diagonal_coefficients=(_q(4), _q(8)),
            cross_terms=(QuadraticCrossTerm(left=0, right=1, coefficient=_q(9)),),
        ),
        source_form=source,
        factor=_q(2),
    )
    with pytest.raises(PydanticCustomError, match="factor times its source"):
        forged.require_scaling_agrees_with_its_sources()


def test_the_scale_envelope_is_published_on_the_request_fields() -> None:
    """The 256-digit factor bound and the capacity envelope are discoverable.

    The factor ceiling is narrower than the shared canonical-rational carrier's
    own 32,768-digit limit, so it is stated on the field rather than left to be
    discovered by trial. The form ceiling is a capacity, so the description says
    the operation refuses it instead of calling it an invalid request.
    """
    factor = QuadraticFormScaleRequest.model_fields["factor"].description
    form = QuadraticFormScaleRequest.model_fields["form"].description
    assert factor is not None
    assert form is not None
    assert str(MAX_QUADRATIC_FORM_COEFFICIENT_DIGITS) in factor
    assert str(MAX_QUADRATIC_SCALE_AXIS) in form
    assert "resource" in form
