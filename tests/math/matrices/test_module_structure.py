"""Matrix module-structure slice (#3727): invariant factors, similarity, centralizers."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
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


def _poly_product(factors) -> list[Fraction]:
    from sympy import Poly, Symbol

    x = Symbol("x")
    product = Poly(1, x)
    for entry in factors:
        factor = entry.factor if hasattr(entry, "factor") else entry
        poly = sum(
            Fraction(term.coefficient.num, term.coefficient.den) * x**term.exponents[0]
            for term in factor.polynomial.terms
        )
        product *= Poly(poly, x)
    return [Fraction(c) for c in product.all_coeffs()[::-1]]


def _monic_to_poly(factor):
    from sympy import Poly, Symbol

    x = Symbol("x")
    poly = sum(
        Fraction(term.coefficient.num, term.coefficient.den) * x**term.exponents[0]
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


def test_centralizer_basis_commutes() -> None:
    from fractions import Fraction

    matrix = _matrix(((2, 0), (0, 3)))
    result = centralizer_basis(matrix)
    entries = [
        [Fraction(v.num, v.den) for v in row] for row in matrix.entries
    ]
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
