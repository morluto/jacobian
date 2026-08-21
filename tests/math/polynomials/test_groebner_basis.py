"""Contract and mathematical tests for the Groebner basis operation."""

from __future__ import annotations

from fractions import Fraction

import sympy

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials._invariants import POLYNOMIAL_INVARIANT_OPERATIONS
from jacobian.math.polynomials._models import PolynomialGroebnerBasisRequest
from jacobian.math.polynomials._operations import polynomial_groebner_basis
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)


def _poly(
    variables: tuple[str, ...],
    terms: dict[tuple[int, ...], int | Fraction],
) -> RationalPolynomial:
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(Fraction(coefficient)),
                    exponents=exponents,
                )
                for exponents, coefficient in sorted(terms.items(), reverse=True)
                if coefficient
            )
        ),
    )


def _poly_to_sympy(poly: RationalPolynomial, variables: tuple[str, ...]) -> sympy.Expr:
    symbols = sympy.symbols(variables)
    return sum(
        sympy.Rational(term.coefficient.num, term.coefficient.den)
        * sympy.prod(sym**exp for sym, exp in zip(symbols, term.exponents, strict=True))
        for term in poly.polynomial.terms
    )


def test_groebner_basis_operation_is_in_catalog() -> None:
    operation_ids = {tool.operation_id for tool in POLYNOMIAL_INVARIANT_OPERATIONS}
    assert "polynomial.ideal.groebner_basis.compute" in operation_ids


def test_groebner_basis_of_principal_ideal_is_generator() -> None:
    request = PolynomialGroebnerBasisRequest(
        generators=(_poly(("x",), {(2,): 1}),),
    )
    result = polynomial_groebner_basis(request)
    assert len(result.basis) == 1
    assert result.basis[0].polynomial.terms[0].exponents == (2,)


def test_groebner_basis_reduces_generators_to_zero() -> None:
    """Every input generator must reduce to zero modulo the Groebner basis."""
    variables = ("x", "y")
    generators = (
        _poly(variables, {(2, 0): 1, (0, 1): -1}),
        _poly(variables, {(1, 1): 1, (0, 0): -1}),
    )
    request = PolynomialGroebnerBasisRequest(
        generators=generators,
        monomial_order="grevlex",
    )
    result = polynomial_groebner_basis(request)
    symbols = sympy.symbols(variables)
    for gen in generators:
        # Use groebner basis to check that generators reduce to zero
        g = sympy.groebner(
            [_poly_to_sympy(b, variables) for b in result.basis],
            *symbols,
            order="grevlex",
        )
        remainder = g.reduce(_poly_to_sympy(gen, variables))
        assert remainder[1] == 0, f"Generator {gen} does not reduce to zero"


def test_groebner_basis_supports_all_monomial_orders() -> None:
    for order in ("lex", "grlex", "grevlex"):
        request = PolynomialGroebnerBasisRequest(
            generators=(_poly(("x", "y"), {(1, 1): 1}),),
            monomial_order=order,
        )
        result = polynomial_groebner_basis(request)
        assert result.monomial_order == order


def test_groebner_basis_result_variables_match_input() -> None:
    variables = ("x", "y", "z")
    request = PolynomialGroebnerBasisRequest(
        generators=(_poly(variables, {(1, 0, 0): 1}),),
    )
    result = polynomial_groebner_basis(request)
    assert result.variables == variables
