"""Polynomial Smith identities and independently computed determinantal divisors."""

from itertools import combinations, pairwise

import pytest
from sympy import QQ, Matrix, Poly, Rational, Symbol, gcd

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.certified_snf import polynomial_smith_decomposition
from jacobian.math.matrices.certified_snf.polynomial import PolynomialSmithDecomposition
from jacobian.math.matrices.symbolic import RationalPolynomialMatrix
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

t = Symbol("t")


def _matrix(
    rows: list[list[object]], *, columns: int | None = None
) -> RationalPolynomialMatrix:
    return RationalPolynomialMatrix(
        variables=("t",),
        row_count=len(rows),
        column_count=len(rows[0]) if rows else (columns or 0),
        entries=tuple(
            tuple(
                RationalPolynomial(
                    variables=("t",),
                    polynomial=SparseRationalPolynomial(
                        terms=tuple(
                            RationalPolynomialTerm(
                                coefficient=CanonicalRational(
                                    num=int(c.p), den=int(c.q)
                                ),
                                exponents=exponents,
                            )
                            for exponents, c in Poly(value, t, domain=QQ).terms()
                            if c
                        )
                    ),
                )
                for value in row
            )
            for row in rows
        ),
    )


def _sympy(matrix: RationalPolynomialMatrix) -> Matrix:
    return Matrix(
        matrix.row_count,
        matrix.column_count,
        [
            sum(
                Rational(term.coefficient.num, term.coefficient.den)
                * t ** term.exponents[0]
                for term in p.polynomial.terms
            )
            for row in matrix.entries
            for p in row
        ],
    )


def _identities(
    source: RationalPolynomialMatrix, result: PolynomialSmithDecomposition
) -> None:
    a, d, u, v = map(
        _sympy,
        (
            source,
            result.diagonal,
            result.left_transformation,
            result.right_transformation,
        ),
    )
    assert (u * a * v).applyfunc(lambda value: value.expand()) == d
    for transformation in (u, v):
        determinant = Poly(transformation.det(), t, domain=QQ)
        assert determinant and determinant.degree() == 0
    factors = [Poly(d[i, i], t, domain=QQ) for i in range(min(a.shape))]
    nonzero = [factor for factor in factors if factor]
    assert all(factor.LC() == 1 for factor in nonzero)
    assert factors[: len(nonzero)] == nonzero
    assert all(right.rem(left).is_zero for left, right in pairwise(nonzero))
    assert all(d[i, j] == 0 for i in range(d.rows) for j in range(d.cols) if i != j)
    product = Poly(1, t, domain=QQ)
    for size, factor in enumerate(nonzero, 1):
        divisor = Poly(0, t, domain=QQ)
        for rows in combinations(range(a.rows), size):
            for columns in combinations(range(a.cols), size):
                divisor = gcd(
                    divisor, Poly(a.extract(rows, columns).det(), t, domain=QQ)
                )
        product *= factor
        assert divisor.monic() == product


@pytest.mark.parametrize(
    "rows,expected",
    [
        ([[t, 1], [0, t]], [1, t * t]),
        ([[t, t * t, 0], [0, 0, 0]], [t, 0]),
        ([[2 * t, Rational(1, 2)], [0, 3 * t]], [1, t * t]),
        ([[0, 0], [t, t * t], [0, 0]], [t, 0]),
        ([[t, t + 1]], [1]),
        ([[t * t + 1, t]], [1]),
        ([[t * t + 1, t * t + t]], [1]),
        ([[1, 1 + t], [t, t * t]], [1, t]),
        ([[t, t + 1], [t + 2, t + 3]], [1, 1]),
        ([[t, t + 1], [2 * t, 2 * t + 2]], [1, 0]),
        ([[t, t], [t, t]], [t, 0]),
        ([[t * t, t * (t + 1)], [t * (t + 2), t * (t + 3)]], [t, t]),
        ([[t + 1, 0], [0, t + 2]], [1, (t + 1) * (t + 2)]),
        ([[t + 1, t + 1], [t + 1, t + 1]], [t + 1, 0]),
    ],
)
def test_motivating_rectangular_and_rational_unit_cases(
    rows: list[list[object]], expected: list[object]
) -> None:
    source = _matrix(rows)
    result = polynomial_smith_decomposition(source)
    _identities(source, result)
    d = _sympy(result.diagonal)
    assert [Poly(d[i, i], t) for i in range(min(d.shape))] == [
        Poly(value, t) for value in expected
    ]
    assert (
        PolynomialSmithDecomposition.model_validate_json(result.model_dump_json())
        == result
    )


@pytest.mark.parametrize(
    "rows,columns", [([], 0), ([], 5), ([[], [], []], 0), ([[0, 0, 0], [0, 0, 0]], 3)]
)
def test_zero_and_empty_shapes(rows: list[list[object]], columns: int) -> None:
    source = _matrix(rows, columns=columns)
    result = polynomial_smith_decomposition(source)
    _identities(source, result)
    assert result.diagonal == source


def test_rational_units_need_not_have_determinant_plus_or_minus_one() -> None:
    source = _matrix([[Rational(2, 3) * t]])
    result = polynomial_smith_decomposition(source)
    _identities(source, result)
    assert _sympy(result.left_transformation).det() == Rational(3, 2)


def test_dense_constant_matrix_uses_field_elimination_bounds() -> None:
    n = 8
    source = _matrix([[1 + int(i == j) for j in range(n)] for i in range(n)])
    result = polynomial_smith_decomposition(source)
    a, d, u, v = map(
        _sympy,
        (
            source,
            result.diagonal,
            result.left_transformation,
            result.right_transformation,
        ),
    )
    assert u * a * v == d == Matrix.eye(n)


def test_excessive_dense_transforms_and_sparse_degree_are_rejected() -> None:
    with pytest.raises(OperationResourceAdmissionError):
        polynomial_smith_decomposition(_matrix([], columns=129))
    with pytest.raises(OperationResourceAdmissionError):
        polynomial_smith_decomposition(_matrix([[t**4096, t + 1]]))


def test_high_degree_monomial_partial_permutation_keeps_sparse_support() -> None:
    source = _matrix(
        [[0, 3 * t**32768, 0], [0, 0, 0], [Rational(2, 7) * t**8192, 0, 0]]
    )
    result = polynomial_smith_decomposition(source)
    d, u, v = map(
        _sympy,
        (result.diagonal, result.left_transformation, result.right_transformation),
    )
    assert (u * _sympy(source) * v).applyfunc(lambda value: value.expand()) == d
    assert [d[i, i] for i in range(3)] == [t**8192, t**32768, 0]


def test_sparse_scalar_polynomial_never_expands_missing_degrees() -> None:
    source = _matrix([[Rational(2, 3) * t**32768 + Rational(3, 7)]])
    result = polynomial_smith_decomposition(source)
    assert _sympy(result.diagonal) == Matrix([[t**32768 + Rational(9, 14)]])
    assert _sympy(result.left_transformation) == Matrix([[Rational(3, 2)]])


def test_serialized_diagonal_is_an_unchanged_polynomial_matrix_input() -> None:
    first = polynomial_smith_decomposition(_matrix([[t, 1], [0, t]]))
    restored = RationalPolynomialMatrix.model_validate_json(
        first.diagonal.model_dump_json()
    )
    second = polynomial_smith_decomposition(restored)
    assert second.diagonal == first.diagonal
    _identities(restored, second)


def test_empty_matrix_at_dense_cell_boundary_needs_only_identity_terms() -> None:
    source = _matrix([], columns=128)
    result = polynomial_smith_decomposition(source)
    assert result.diagonal == source
    assert _sympy(result.right_transformation) == Matrix.eye(128)


def test_sparse_coefficients_exceeding_total_integer_storage_are_rejected() -> None:
    coefficient = CanonicalRational(num=2**2200 + 1, den=1)
    polynomial = RationalPolynomial(
        variables=("t",),
        polynomial=SparseRationalPolynomial(
            terms=(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational(num=1, den=1), exponents=(4095,)
                ),
                *(
                    RationalPolynomialTerm(coefficient=coefficient, exponents=(degree,))
                    for degree in range(4094, -1, -1)
                ),
            )
        ),
    )
    source = RationalPolynomialMatrix(
        variables=("t",), row_count=1, column_count=1, entries=((polynomial,),)
    )
    with pytest.raises(OperationResourceAdmissionError, match="coefficient storage"):
        polynomial_smith_decomposition(source)


def test_proportional_sparse_polynomial_entries_retain_the_common_factor() -> None:
    p = t**32768 + 1
    source = _matrix([[2 * p, p], [p, 3 * p]])
    result = polynomial_smith_decomposition(source)
    d, u, v = map(
        _sympy,
        (result.diagonal, result.left_transformation, result.right_transformation),
    )
    assert (u * _sympy(source) * v).applyfunc(lambda value: value.expand()) == d
    assert d == Matrix.diag(p, p)


def test_copied_inconsistent_axes_are_rejected_before_admission() -> None:
    source = _matrix([[t]])
    forged = source.model_copy(update={"row_count": 0})
    with pytest.raises(OperationDomainValidationError, match="structural"):
        polynomial_smith_decomposition(forged)
