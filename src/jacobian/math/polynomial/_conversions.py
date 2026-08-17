"""Exact conversions between polynomial contracts and SymPy ``Poly`` values."""

from __future__ import annotations

from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_TERMS,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

__all__ = [
    "rational_from_sympy",
    "rational_polynomial_from_sympy",
    "rational_polynomial_to_sympy",
    "symbols_for_variables",
]


def symbols_for_variables(variables: tuple[str, ...]) -> tuple[Any, ...]:
    """Return SymPy generators in the authoritative declared order."""

    from sympy import Symbol

    return tuple(Symbol(variable) for variable in variables)


def rational_from_sympy(value: Any) -> CanonicalRational:
    """Convert one exact SymPy rational, rejecting floats and symbolic values."""

    from sympy import Rational

    if not isinstance(value, Rational):
        raise ValueError("SymPy value is not an exact rational")
    return CanonicalRational.from_integer_ratio(int(value.p), int(value.q))


def rational_polynomial_to_sympy(polynomial: RationalPolynomial) -> Any:
    """Convert the exact sparse contract into a QQ ``Poly`` without reordering."""

    from sympy import QQ, Poly, Rational

    return Poly.from_dict(
        {
            term.exponents: Rational(*term.coefficient.as_integer_ratio())
            for term in polynomial.polynomial.terms
        },
        *symbols_for_variables(polynomial.variables),
        domain=QQ,
    )


def rational_polynomial_from_sympy(
    polynomial: Any,
    variables: tuple[str, ...],
    *,
    maximum_terms: int = MAX_POLYNOMIAL_TERMS,
) -> RationalPolynomial:
    """Convert a QQ ``Poly`` back to the canonical sparse contract."""

    from sympy import QQ

    if polynomial.domain != QQ:
        raise ValueError("SymPy polynomial must have the exact QQ domain")
    if tuple(str(generator) for generator in polynomial.gens) != variables:
        raise ValueError("SymPy polynomial generators do not match the declared order")
    terms = tuple(
        (tuple(int(exponent) for exponent in exponents), coefficient)
        for exponents, coefficient in polynomial.terms()
        if coefficient != 0
    )
    if len(terms) > maximum_terms:
        raise ValueError(
            f"polynomial result exceeds the {maximum_terms}-term operation budget"
        )
    return RationalPolynomial(
        variables=variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=rational_from_sympy(coefficient),
                    exponents=exponents,
                )
                for exponents, coefficient in terms
            )
        ),
    )
