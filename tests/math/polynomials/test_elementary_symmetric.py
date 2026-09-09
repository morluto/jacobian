"""Canonical elementary symmetric families."""

from jacobian.math.polynomials._elementary_symmetric import (
    ElementarySymmetricFamilyRequest,
    elementary_symmetric_family,
)
from jacobian.math.polynomials.values import RationalPolynomial


def support(polynomial: RationalPolynomial) -> tuple[tuple[int, ...], ...]:
    return tuple(term.exponents for term in polynomial.polynomial.terms)


def test_family_through_degree_two() -> None:
    result = elementary_symmetric_family(
        ElementarySymmetricFamilyRequest(variables=("x", "y", "z"), maximum_degree=2)
    )
    assert support(result.polynomials[0]) == ((0, 0, 0),)
    assert support(result.polynomials[1]) == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert support(result.polynomials[2]) == ((1, 1, 0), (1, 0, 1), (0, 1, 1))


def test_empty_axis_e_zero_is_a_canonical_constant() -> None:
    result = elementary_symmetric_family(
        ElementarySymmetricFamilyRequest(variables=(), maximum_degree=0)
    )
    assert result.polynomials[0].variables == ()
    assert support(result.polynomials[0]) == ((),)
