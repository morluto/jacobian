from __future__ import annotations

from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    FormRequest,
)
from jacobian.math.number_theory.quadratic_forms.general._tools import (
    compute_diagonalization,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalQuadraticForm,
)


def _r(value: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational.from_integer_ratio(value, denominator)


def _determinant(matrix: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    """Small independent Leibniz-free Laplace oracle for dimensions <= 3."""
    if not matrix:
        return Fraction(1)
    if len(matrix) == 1:
        return matrix[0][0]
    return sum(
        (
            (-1) ** column
            * matrix[0][column]
            * _determinant(
                tuple(
                    tuple(value for j, value in enumerate(row) if j != column)
                    for row in matrix[1:]
                )
            )
            for column in range(len(matrix))
        ),
        Fraction(),
    )


def _source_matrix(form: RationalQuadraticForm) -> tuple[tuple[Fraction, ...], ...]:
    n = len(form.axis)
    matrix = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for index, coefficient in enumerate(form.diagonal_coefficients):
        matrix[index][index] = coefficient.as_fraction()
    for term in form.cross_terms:
        matrix[term.left][term.right] = term.coefficient.as_fraction() / 2
        matrix[term.right][term.left] = term.coefficient.as_fraction() / 2
    return tuple(tuple(row) for row in matrix)


@pytest.mark.parametrize(
    ("form", "expected_rank"),
    (
        (
            RationalQuadraticForm(
                axis=("x", "y"), diagonal_coefficients=(_r(0), _r(0))
            ),
            0,
        ),
        (
            RationalQuadraticForm(
                axis=("x", "y"), diagonal_coefficients=(_r(1), _r(0))
            ),
            1,
        ),
        (
            RationalQuadraticForm(
                axis=("x", "y"),
                diagonal_coefficients=(_r(2), _r(3)),
                cross_terms=(QuadraticCrossTerm(left=0, right=1, coefficient=_r(1)),),
            ),
            2,
        ),
        (
            RationalQuadraticForm(
                axis=("x", "y", "z"),
                diagonal_coefficients=(_r(0), _r(0), _r(0)),
                cross_terms=(
                    QuadraticCrossTerm(left=0, right=1, coefficient=_r(2)),
                    QuadraticCrossTerm(left=1, right=2, coefficient=_r(4)),
                ),
            ),
            2,
        ),
    ),
)
def test_diagonalization_matches_independent_rational_matrix_oracle(
    form: RationalQuadraticForm, expected_rank: int
) -> None:
    result = compute_diagonalization(FormRequest(form=form))
    source = _source_matrix(form)
    change = tuple(
        tuple(value.as_fraction() for value in row) for row in result.change.entries
    )
    diagonal = tuple(value.as_fraction() for value in result.diagonal)
    reconstructed = tuple(
        tuple(
            sum(
                (
                    change[k][i] * source[k][ell] * change[ell][j]
                    for k in range(len(form.axis))
                    for ell in range(len(form.axis))
                ),
                Fraction(),
            )
            for j in range(len(form.axis))
        )
        for i in range(len(form.axis))
    )
    expected = tuple(
        tuple(diagonal[i] if i == j else Fraction(0) for j in range(len(form.axis)))
        for i in range(len(form.axis))
    )

    assert reconstructed == expected
    assert _determinant(change) != 0
    assert sum(value != 0 for value in diagonal) == expected_rank
    assert result.source_axis == form.axis
    assert result.basis_axis == tuple(f"basis_{i}" for i in range(len(form.axis)))
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_diagonalization_admits_diagonal_large_height_without_elimination() -> None:
    form = RationalQuadraticForm(
        axis=("x", "y"),
        diagonal_coefficients=(_r(10**200), _r(0)),
    )
    result = compute_diagonalization(FormRequest(form=form))
    assert result.diagonal == form.diagonal_coefficients
    assert result.change.entries == ((_r(1), _r(0)), (_r(0), _r(1)))


def test_diagonalization_accepts_exact_dimension_and_work_boundary() -> None:
    axis = tuple(f"x{i}" for i in range(64))
    form = RationalQuadraticForm(
        axis=axis, diagonal_coefficients=tuple(_r(1) for _ in axis)
    )
    result = compute_diagonalization(FormRequest(form=form))
    assert result.source_axis == axis
    assert result.basis_axis == tuple(f"basis_{i}" for i in range(64))
    assert result.change.row_count == result.change.column_count == 64


@pytest.mark.parametrize(
    "axis_size",
    (65,),
)
def test_diagonalization_rejects_dimension_before_kernel(
    axis_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jacobian.math.number_theory.quadratic_forms.general._tools as tools

    form = RationalQuadraticForm(
        axis=tuple(f"x{i}" for i in range(axis_size)),
        diagonal_coefficients=tuple(_r(1) for _ in range(axis_size)),
    )
    monkeypatch.setattr(
        tools,
        "quadratic_diagonalization",
        lambda _: pytest.fail("elimination kernel ran before dimension rejection"),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        compute_diagonalization(FormRequest(form=form))
    assert (
        error.value.errors()[0]["type"] == "quadratic_form.diagonalization_axis_bound"
    )


def test_diagonalization_rejects_growth_before_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.number_theory.quadratic_forms.general._tools as tools

    form = RationalQuadraticForm(
        axis=tuple(f"x{i}" for i in range(64)),
        diagonal_coefficients=tuple(_r(1) for _ in range(64)),
        cross_terms=(QuadraticCrossTerm(left=0, right=1, coefficient=_r(1)),),
    )
    monkeypatch.setattr(
        tools,
        "quadratic_diagonalization",
        lambda _: pytest.fail("elimination kernel ran before growth rejection"),
    )
    with pytest.raises(OperationResourceAdmissionError) as error:
        compute_diagonalization(FormRequest(form=form))
    assert error.value.errors()[0]["type"] == (
        "quadratic_form.diagonalization_coefficient_growth"
    )
