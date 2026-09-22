from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.matrices.values import RationalMatrix
from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    FormRequest,
    PullbackRequest,
)
from jacobian.math.number_theory.quadratic_forms.general._tools import (
    compute_diagonalization,
    compute_pullback,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalQuadraticForm,
)


def _r(value: int) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(value, 1)


def test_pair_pivot_diagonalization_reconstructs_coupled_form() -> None:
    form = RationalQuadraticForm(
        axis=("x", "y", "z"),
        diagonal_coefficients=(_r(0), _r(0), _r(0)),
        cross_terms=(
            QuadraticCrossTerm(left=0, right=1, coefficient=_r(2)),
            QuadraticCrossTerm(left=1, right=2, coefficient=_r(4)),
        ),
    )
    result = compute_diagonalization(FormRequest(form=form))
    source = (
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(2)),
        (Fraction(0), Fraction(2), Fraction(0)),
    )
    change = tuple(
        tuple(entry.as_fraction() for entry in row) for row in result.change.entries
    )
    reconstructed = tuple(
        tuple(
            sum(
                change[k][i] * source[k][ell] * change[ell][j]
                for k in range(3)
                for ell in range(3)
            )
            for j in range(3)
        )
        for i in range(3)
    )
    expected = tuple(
        tuple(
            result.diagonal[i].as_fraction() if i == j else Fraction(0)
            for j in range(3)
        )
        for i in range(3)
    )
    assert reconstructed == expected
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored.form.axis == form.axis
    assert restored.change == result.change
    assert restored.diagonal == result.diagonal


def test_pullback_keeps_target_axis_and_matrix_shape() -> None:
    form = RationalQuadraticForm(
        axis=("x", "y"),
        diagonal_coefficients=(_r(1), _r(2)),
    )
    matrix = RationalMatrix(
        row_count=2,
        column_count=1,
        entries=((_r(1),), (_r(1),)),
    )
    result = compute_pullback(
        PullbackRequest(form=form, matrix=matrix, target_axis=("u",))
    )
    assert result.form.axis == ("u",)
    assert result.form.diagonal_coefficients == (_r(3),)
    assert result.source_axis == ("x", "y")


def test_pullback_rejects_target_dimension_before_dense_output() -> None:
    form = RationalQuadraticForm(axis=("x",), diagonal_coefficients=(_r(1),))
    target_axis = tuple(f"u{i}" for i in range(129))
    matrix = RationalMatrix(
        row_count=1,
        column_count=129,
        entries=(tuple(_r(1) for _ in target_axis),),
    )
    with pytest.raises(OperationResourceAdmissionError):
        compute_pullback(
            PullbackRequest(form=form, matrix=matrix, target_axis=target_axis)
        )


def test_pullback_rejects_coefficient_growth_before_fraction_products() -> None:
    form = RationalQuadraticForm(axis=("x",), diagonal_coefficients=(_r(1),))
    huge = _r(10**299)
    matrix = RationalMatrix(
        row_count=1,
        column_count=1,
        entries=((huge,),),
    )

    with pytest.raises(OperationResourceAdmissionError) as error:
        compute_pullback(PullbackRequest(form=form, matrix=matrix, target_axis=("u",)))
    assert error.value.errors()[0]["type"] == (
        "quadratic_form.pullback_coefficient_growth"
    )


def test_quadratic_diagonalization_zero_dimensional_form() -> None:
    result = compute_diagonalization(
        FormRequest(form=RationalQuadraticForm(axis=(), diagonal_coefficients=()))
    )
    assert result.diagonal == ()
    assert result.change.row_count == result.change.column_count == 0
