"""Exact coverage, singular fibers and independent spectral evidence."""

import json
from fractions import Fraction
from itertools import pairwise
from typing import Any

import pytest
import sympy as sp
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.analysis.intervals import ClosedRationalInterval
from jacobian.math.matrices.inertia_cells import (
    InertiaCellsResult,
    InertiaOpenCell,
    InertiaPointCell,
    compute_inertia_cells,
)
from jacobian.math.matrices.symbolic.values import RationalPolynomialMatrix
from jacobian.math.polynomials._conversions import rational_polynomial_from_sympy

T = sp.Symbol("t")


def _source(matrix: Any) -> RationalPolynomialMatrix:
    return RationalPolynomialMatrix(
        variables=("t",),
        row_count=matrix.rows,
        column_count=matrix.cols,
        entries=tuple(
            tuple(
                rational_polynomial_from_sympy(
                    sp.Poly(matrix[i, j], T, domain=sp.QQ), ("t",)
                )
                for j in range(matrix.cols)
            )
            for i in range(matrix.rows)
        ),
    )


def _interval(a: int | Fraction = -2, b: int | Fraction = 2) -> ClosedRationalInterval:
    return ClosedRationalInterval(
        lower=CanonicalRational.from_fraction(Fraction(a)),
        upper=CanonicalRational.from_fraction(Fraction(b)),
    )


def _value(boundary: Any) -> Any:
    value = boundary.value
    if isinstance(value, CanonicalRational):
        return sp.Rational(value.num, value.den)
    return sp.CRootOf(sp.Poly.from_list(value.polynomial, T), value.real_root_index)


def _counts(matrix: Any) -> tuple[int, int, int]:
    positive = negative = zero = 0
    for eigenvalue, multiplicity in matrix.eigenvals().items():
        if eigenvalue == 0:
            zero += multiplicity
        elif eigenvalue.is_positive:
            positive += multiplicity
        else:
            assert eigenvalue.is_negative
            negative += multiplicity
    return positive, negative, zero


def _check(
    matrix: Any, interval: ClosedRationalInterval | None = None
) -> InertiaCellsResult:
    interval = interval or _interval()
    result = compute_inertia_cells(_source(matrix), interval)
    points = [cell for cell in result.cells if isinstance(cell, InertiaPointCell)]
    values = [_value(p.parameter) for p in points]
    assert values[0] == sp.Rational(interval.lower.num, interval.lower.den)
    assert values[-1] == sp.Rational(interval.upper.num, interval.upper.den)
    assert all(a < b for a, b in pairwise(values))
    for cell in result.cells:
        if isinstance(cell, InertiaPointCell):
            parameter = _value(cell.parameter)
        else:
            assert isinstance(cell, InertiaOpenCell)
            lo = cell.lower.isolating_interval.upper.as_fraction()
            hi = cell.upper.isolating_interval.lower.as_fraction()
            parameter = sp.Rational((lo + hi) / 2)
            assert _value(cell.lower) < parameter < _value(cell.upper)
        expected = _counts(matrix.subs(T, parameter))
        assert (cell.n_positive, cell.n_negative, cell.n_zero) == expected
    assert InertiaCellsResult.model_validate_json(result.model_dump_json()) == result
    return result


def test_persistent_nullspace_and_irrational_rank_drops() -> None:
    result = _check(sp.diag(T**2 - 2, 0))
    assert len(result.cells) == 7
    assert [c.n_zero for c in result.cells] == [1, 1, 2, 1, 2, 1, 1]


def test_tangency_does_not_disappear() -> None:
    result = _check(sp.diag(T**2, 1))
    assert len(result.cells) == 5
    assert result.cells[2].n_zero == 1
    assert all(c.n_negative == 0 for c in result.cells)


def test_indefinite_family_without_real_rank_drops() -> None:
    result = _check(sp.Matrix([[T, 1], [1, -T]]))
    assert len(result.cells) == 3
    assert all(
        (c.n_positive, c.n_negative, c.n_zero) == (1, 1, 0) for c in result.cells
    )


@pytest.mark.parametrize(
    "matrix", [sp.zeros(0), sp.zeros(3), sp.diag(1, -1, 0), sp.Matrix([[0, 1], [1, 0]])]
)
def test_empty_zero_and_constant(matrix: Any) -> None:
    _check(matrix)
    _check(matrix, _interval(1, 1))


def test_endpoint_singularities_and_singleton() -> None:
    matrix = sp.diag(T * (T - 1), T)
    assert len(_check(matrix, _interval(0, 1)).cells) == 3
    assert _check(matrix, _interval(0, 0)).cells[0].n_zero == 2


def test_repeated_roots_and_distinct_factor_union() -> None:
    _check(sp.diag((T**2 - 2) ** 2, T**2 - 2, T**2 - 3))


def test_congruence_preserves_cell_inertia() -> None:
    transform = sp.Matrix([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    diagonal = sp.diag(T, T - 1, 0)
    source = transform.T * diagonal * transform
    result = compute_inertia_cells(_source(source), _interval())
    for cell in result.cells:
        if isinstance(cell, InertiaPointCell):
            value = _value(cell.parameter)
        else:
            value = sp.Rational(
                (
                    cell.lower.isolating_interval.upper.as_fraction()
                    + cell.upper.isolating_interval.lower.as_fraction()
                )
                / 2
            )
        assert (cell.n_positive, cell.n_negative, cell.n_zero) == _counts(
            diagonal.subs(T, value)
        )


def test_large_repeated_singular_blocks() -> None:
    matrix = sp.diag(*([T**2 - 2] * 64), *([0] * 64))
    result = compute_inertia_cells(_source(matrix), _interval())
    assert len(result.cells) == 7
    assert [c.n_zero for c in result.cells] == [64, 64, 128, 64, 128, 64, 64]


def test_polynomial_symmetry_rejection() -> None:
    with pytest.raises(OperationDomainValidationError, match="symmetric"):
        compute_inertia_cells(_source(sp.Matrix([[T, 1], [0, T]])), _interval())


def test_connected_algebraic_degree_rejection() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="degree-16"):
        compute_inertia_cells(_source(sp.Matrix([[T**17 - T - 1]])), _interval())


def test_excessive_coefficient_growth_rejection() -> None:
    with pytest.raises(OperationResourceAdmissionError, match="carrier"):
        compute_inertia_cells(
            _source(sp.Matrix([[T**2 - (10**1000 + 7)]])),
            _interval(-(10**501), 10**501),
        )


def test_structural_cell_contradiction_rejected() -> None:
    result = compute_inertia_cells(_source(sp.diag(T)), _interval())
    payload = result.model_dump(mode="json")
    payload["cells"][1]["n_zero"] = 1
    with pytest.raises(ValidationError, match="sum"):
        InertiaCellsResult.model_validate_json(json.dumps(payload))


def test_singular_endpoint_samples_stay_in_open_cells() -> None:
    _check(sp.diag((T + 2) * (T**2 - 2), 0))
    _check(sp.diag(T**2 - 2), _interval(Fraction(-3, 2), Fraction(-7, 5)))


def test_algebraic_coefficient_sign_at_nonlinear_boundary() -> None:
    transform = sp.Matrix([[1, 1], [1, 2]])
    diagonal = sp.diag(T**3 - T - 1, T + 2)
    source = transform.T * diagonal * transform
    result = compute_inertia_cells(_source(source), _interval())
    for cell in result.cells:
        if isinstance(cell, InertiaPointCell):
            value = _value(cell.parameter)
        else:
            value = sp.Rational(
                (
                    cell.lower.isolating_interval.upper.as_fraction()
                    + cell.upper.isolating_interval.lower.as_fraction()
                )
                / 2
            )
        assert (cell.n_positive, cell.n_negative, cell.n_zero) == _counts(
            diagonal.subs(T, value)
        )


def test_positive_denominator_scaling_preserves_profile() -> None:
    transform = sp.Matrix([[1, 1], [1, 2]])
    matrix = transform.T * sp.diag(T**2 - 2, 0) * transform
    result = compute_inertia_cells(
        _source(matrix / sp.Integer(2) ** 20000), _interval()
    )
    ordinary = compute_inertia_cells(_source(matrix), _interval())
    assert result.cells == ordinary.cells


def test_affine_large_coefficient_and_high_degree_singleton() -> None:
    affine = _check(sp.Matrix([[T + 10**1000]]))
    assert len(affine.cells) == 3
    assert all(c.n_positive == 1 for c in affine.cells)
    singleton = _check(sp.Matrix([[T**17]]), _interval(0, 0))
    assert len(singleton.cells) == 1
    assert singleton.cells[0].n_zero == 1


def test_large_rational_affine_transition_inside_domain() -> None:
    result = _check(sp.Matrix([[10**1000 * T - 1]]))
    assert len(result.cells) == 5
    assert result.cells[2].n_zero == 1


def test_singleton_specialization_of_high_degree_nondiagonal_matrix() -> None:
    result = _check(sp.Matrix([[T**17, 1], [1, -(T**17)]]), _interval(0, 0))
    assert (
        result.cells[0].n_positive,
        result.cells[0].n_negative,
        result.cells[0].n_zero,
    ) == (1, 1, 0)
