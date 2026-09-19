"""Matrix module-structure slice (#3727): invariant factors, similarity, centralizers."""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import Any

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.matrices.canonical_forms.operations import (
    centralizer_basis,
    decide_similarity,
    invariant_factor_profile,
)
from jacobian.math.matrices.values import RationalMatrix


def _matrix(rows: tuple[tuple[int, ...], ...]) -> RationalMatrix:
    return RationalMatrix(
        domain="QQ",
        row_count=len(rows),
        column_count=len(rows[0]),
        entries=tuple(
            tuple(CanonicalRational.from_fraction(Fraction(v)) for v in row)
            for row in rows
        ),
    )


def _poly_product(factors: Iterable[Any]) -> list[Fraction]:
    from sympy import Poly, Symbol

    x = Symbol("x")
    product = Poly(1, x)
    for entry in factors:
        factor = entry.factor if hasattr(entry, "factor") else entry
        poly = sum(
            Fraction(term.coefficient.num, term.coefficient.den)
            * x ** term.exponents[0]
            for term in factor.polynomial.terms
        )
        product *= Poly(poly, x)
    return [Fraction(c) for c in product.all_coeffs()[::-1]]


def _monic_to_poly(factor: Any) -> Any:
    from sympy import Poly, Symbol

    x = Symbol("x")
    poly = sum(
        Fraction(term.coefficient.num, term.coefficient.den) * x ** term.exponents[0]
        for term in factor.polynomial.terms
    )
    return Poly(poly, x)


def test_invariant_factors_divisibility_and_product() -> None:
    from sympy import QQ, Poly, Symbol

    matrix = _matrix(((2, 0), (0, 3)))
    profile = invariant_factor_profile(matrix)
    assert len(profile.invariant_factors) == 1
    assert profile.invariant_factors[-1].factor == profile.minimal_polynomial
    # Product of invariant factors equals the characteristic polynomial.
    x = Symbol("x")
    product = Poly(1, x, domain=QQ)
    for entry in profile.invariant_factors:
        product *= _monic_to_poly(entry.factor)
    char_poly = _monic_to_poly(profile.characteristic_polynomial)
    assert product.as_expr() == char_poly.as_expr()


def test_similarity_permutation_and_nonsimilar() -> None:
    left = _matrix(((2, 0), (0, 3)))
    permuted = _matrix(((3, 0), (0, 2)))
    assert decide_similarity(left, permuted).similar is True
    jordan = _matrix(((2, 1), (0, 2)))
    assert decide_similarity(left, jordan).similar is False
    scalar = _matrix(((2, 0), (0, 2)))
    assert decide_similarity(left, scalar).similar is False


def test_centralizer_dimensions() -> None:
    # Distinct eigenvalues: polynomials in A, dimension 2.
    assert centralizer_basis(_matrix(((2, 0), (0, 3)))).dimension == 2
    # Scalar matrix: full M_2, dimension 4.
    assert centralizer_basis(_matrix(((2, 0), (0, 2)))).dimension == 4
    # Nontrivial Jordan block: dimension 2.
    assert centralizer_basis(_matrix(((2, 1), (0, 2)))).dimension == 2
    # Equal diagonal entries form a full matrix block, independently of the
    # distinct third eigenspace.
    assert centralizer_basis(_matrix(((2, 0, 0), (0, 2, 0), (0, 0, 3)))).dimension == 5


def test_centralizer_two_by_two_maximum_height_uses_i_a_basis() -> None:
    scalar = 10**255
    matrix = _matrix(((scalar, scalar - 1), (0, scalar)))
    result = centralizer_basis(matrix)

    assert result.dimension == 2
    assert len(result.basis) == 2
    assert result.basis[0] == _matrix(((1, 0), (0, 1)))
    assert result.basis[1] == matrix
    # The nonzero off-diagonal entry proves I and A are linearly independent.
    assert result.basis[1].entries[0][1].num == scalar - 1
    _assert_centralizer_commutes(matrix, result.basis)


def test_centralizer_basis_commutes() -> None:
    from fractions import Fraction

    matrix = _matrix(((2, 0), (0, 3)))
    result = centralizer_basis(matrix)
    entries = [[Fraction(v.num, v.den) for v in row] for row in matrix.entries]
    for basis_matrix in result.basis:
        other = [[Fraction(v.num, v.den) for v in row] for row in basis_matrix.entries]
        left = [
            [sum(entries[i][k] * other[k][j] for k in range(2)) for j in range(2)]
            for i in range(2)
        ]
        right = [
            [sum(other[i][k] * entries[k][j] for k in range(2)) for j in range(2)]
            for i in range(2)
        ]
        assert left == right


def _assert_centralizer_commutes(
    matrix: RationalMatrix, basis: Iterable[RationalMatrix]
) -> None:
    source = [
        [Fraction(value.num, value.den) for value in row] for row in matrix.entries
    ]
    for basis_matrix in basis:
        other = [
            [Fraction(value.num, value.den) for value in row]
            for row in basis_matrix.entries
        ]
        assert _multiply(source, other) == _multiply(other, source)


def test_centralizer_nine_by_nine_jordan_kernel_is_exact() -> None:
    size = 9
    matrix = _matrix(
        tuple(
            tuple(1 if column == row + 1 else 0 for column in range(size))
            for row in range(size)
        )
    )
    result = centralizer_basis(matrix)

    assert result.dimension == size
    assert len(result.basis) == size
    _assert_centralizer_commutes(matrix, result.basis)


def test_centralizer_sixteen_by_sixteen_jordan_kernel_is_exact() -> None:
    size = 16
    matrix = _matrix(
        tuple(
            tuple(1 if column == row + 1 else 0 for column in range(size))
            for row in range(size)
        )
    )
    result = centralizer_basis(matrix)

    assert result.dimension == size
    assert len(result.basis) == size
    _assert_centralizer_commutes(matrix, result.basis)


def test_centralizer_sixteen_by_sixteen_scalar_reaches_maximum_output() -> None:
    size = 16
    scalar = 10**255
    matrix = _matrix(
        tuple(
            tuple(scalar if row == column else 0 for column in range(size))
            for row in range(size)
        )
    )
    result = centralizer_basis(matrix)

    assert result.dimension == size * size
    assert len(result.basis) == size * size
    _assert_centralizer_commutes(matrix, result.basis)


def test_centralizer_general_maximum_height_is_admitted_before_flint() -> None:
    size = 16
    scalar = 10**255
    matrix = _matrix(
        tuple(
            tuple(
                scalar
                if row == column
                else scalar - 1
                if column == (row + 1) % size
                else 0
                for column in range(size)
            )
            for row in range(size)
        )
    )

    with pytest.raises(OperationResourceAdmissionError, match="exact RREF work"):
        centralizer_basis(matrix)


def _multiply(
    left: list[list[Fraction]], right: list[list[Fraction]]
) -> list[list[Fraction]]:
    size = len(left)
    return [
        [
            sum(
                (left[i][k] * right[k][j] for k in range(size)),
                Fraction(0),
            )
            for j in range(size)
        ]
        for i in range(size)
    ]


def test_similar_pair_exhibits_an_explicit_conjugation() -> None:
    """P^-1 A P = B for the swap permutation, matching the verdict."""
    left = [[Fraction(2), Fraction(0)], [Fraction(0), Fraction(3)]]
    swap = [[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]]
    conjugated = _multiply(_multiply(swap, left), swap)
    assert conjugated == [[Fraction(3), Fraction(0)], [Fraction(0), Fraction(2)]]
    assert (
        decide_similarity(_matrix(((2, 0), (0, 3))), _matrix(((3, 0), (0, 2)))).similar
        is True
    )


def test_same_characteristic_polynomial_does_not_imply_similarity() -> None:
    """diag(2,2,3) and diag(J2(2),[3]) share charpoly but not minpoly."""
    diagonal = _matrix(((2, 0, 0), (0, 2, 0), (0, 0, 3)))
    defective = _matrix(((2, 1, 0), (0, 2, 0), (0, 0, 3)))
    assert decide_similarity(diagonal, defective).similar is False
    assert len(invariant_factor_profile(diagonal).invariant_factors) == 2
    assert len(invariant_factor_profile(defective).invariant_factors) == 1


def test_nilpotent_jordan_block_centralizer_is_polynomial() -> None:
    result = centralizer_basis(_matrix(((0, 1, 0), (0, 0, 1), (0, 0, 0))))
    assert result.dimension == 3
    assert len(result.basis) == 3
