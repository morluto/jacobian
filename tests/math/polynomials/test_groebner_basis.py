"""Contract and mathematical tests for the Groebner basis operation."""

from __future__ import annotations

from fractions import Fraction

import sympy

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials._invariants import POLYNOMIAL_INVARIANT_OPERATIONS
from jacobian.math.polynomials._models import (
    PolynomialGroebnerBasisRequest,
    PolynomialGroebnerBudget,
)
from jacobian.math.polynomials._operations import polynomial_groebner_basis
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialIdeal,
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


def _ideal(
    variables: tuple[str, ...],
    *polys: RationalPolynomial,
) -> RationalPolynomialIdeal:
    return RationalPolynomialIdeal(variables=variables, generators=polys)


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
    ideal = _ideal(("x",), _poly(("x",), {(2,): 1}))
    request = PolynomialGroebnerBasisRequest(ideal=ideal)
    result = polynomial_groebner_basis(request)
    assert result.outcome == "COMPUTED"
    assert result.basis is not None
    assert len(result.basis.generators) == 1
    assert result.basis.generators[0].polynomial.terms[0].exponents == (2,)
    assert result.ideal == ideal
    assert result.basis.variables == ("x",)


def test_groebner_basis_reduces_generators_to_zero() -> None:
    """Every input generator must reduce to zero modulo the Groebner basis."""
    variables = ("x", "y")
    generators = (
        _poly(variables, {(2, 0): 1, (0, 1): -1}),
        _poly(variables, {(1, 1): 1, (0, 0): -1}),
    )
    ideal = _ideal(variables, *generators)
    request = PolynomialGroebnerBasisRequest(
        ideal=ideal,
        monomial_order="grevlex",
    )
    result = polynomial_groebner_basis(request)
    assert result.outcome == "COMPUTED"
    assert result.basis is not None
    symbols = sympy.symbols(variables)
    for gen in generators:
        g = sympy.groebner(
            [_poly_to_sympy(b, variables) for b in result.basis.generators],
            *symbols,
            order="grevlex",
        )
        remainder = g.reduce(_poly_to_sympy(gen, variables))
        assert remainder[1] == 0, f"Generator {gen} does not reduce to zero"


def test_groebner_basis_supports_all_monomial_orders() -> None:
    for order in ("lex", "grlex", "grevlex"):
        ideal = _ideal(("x", "y"), _poly(("x", "y"), {(1, 1): 1}))
        request = PolynomialGroebnerBasisRequest(
            ideal=ideal,
            monomial_order=order,
        )
        result = polynomial_groebner_basis(request)
        assert result.outcome == "COMPUTED"
        assert result.monomial_order == order
        assert result.basis is not None


def test_groebner_basis_result_variables_match_input() -> None:
    variables = ("x", "y", "z")
    ideal = _ideal(variables, _poly(variables, {(1, 0, 0): 1}))
    request = PolynomialGroebnerBasisRequest(ideal=ideal)
    result = polynomial_groebner_basis(request)
    assert result.outcome == "COMPUTED"
    assert result.basis is not None
    assert result.basis.variables == variables
    assert result.ideal.variables == variables


def test_groebner_basis_output_budget_returns_typed_outcome() -> None:
    """A valid request that exceeds the caller-selectable output limits must return LIMIT_EXCEEDED."""
    variables = ("x", "y")
    ideal = _ideal(
        variables,
        _poly(variables, {(1, 0): 1}),
        _poly(variables, {(0, 1): 1}),
    )
    request = PolynomialGroebnerBasisRequest(
        ideal=ideal,
        resource_budget=PolynomialGroebnerBudget(
            wall_seconds=10,
            maximum_basis_polynomials=1,
            maximum_output_terms=1024,
        ),
    )
    result = polynomial_groebner_basis(request)
    assert result.outcome == "LIMIT_EXCEEDED"
    assert result.basis is None
    assert result.detail is not None
    assert "polynomial-count" in result.detail


def test_groebner_basis_uses_canonical_ideal_value() -> None:
    """The operation accepts and returns the canonical RationalPolynomialIdeal value."""
    variables = ("x", "y")
    ideal = _ideal(variables, _poly(variables, {(2, 0): 1, (0, 1): -1}))
    request = PolynomialGroebnerBasisRequest(ideal=ideal, monomial_order="grevlex")
    result = polynomial_groebner_basis(request)
    assert result.outcome == "COMPUTED"
    assert result.basis is not None
    # Serialized ideal can be supplied unchanged to the next call.
    round_trip = PolynomialGroebnerBasisRequest(
        ideal=result.basis,
        monomial_order="grevlex",
    )
    second = polynomial_groebner_basis(round_trip)
    assert second.outcome == "COMPUTED"
    assert second.basis is not None
    # Basis ideal retains the same ordered ring.
    assert second.basis.variables == variables
