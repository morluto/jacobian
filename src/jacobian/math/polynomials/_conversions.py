"""Exact conversions between polynomial contracts and SymPy ``Poly`` values."""

from __future__ import annotations

from typing import Any

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_TERMS,
    RationalFunction,
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

__all__ = [
    "rational_from_sympy",
    "rational_function_from_sympy",
    "rational_function_to_sympy",
    "rational_polynomial_from_sympy",
    "rational_polynomial_to_sympy",
    "sparse_rational_polynomial_from_sympy",
    "sparse_rational_polynomial_to_sympy",
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


def sparse_rational_polynomial_to_sympy(
    polynomial: SparseRationalPolynomial,
    variables: tuple[str, ...],
    *,
    symbols: tuple[Any, ...] | None = None,
    cache: dict[tuple[tuple[str, ...], SparseRationalPolynomial], Any] | None = None,
) -> Any:
    """Construct a QQ ``Poly`` from validated sparse data.

    Callers doing several conversions in one request may supply their local
    generators and conversion dictionary. Nothing is retained across requests.
    """

    from sympy import QQ, Poly, Rational

    if not variables:
        raise ValueError("SymPy Poly requires at least one generator")
    key = (variables, polynomial)
    if cache is not None and key in cache:
        return cache[key]
    generators = symbols_for_variables(variables) if symbols is None else symbols
    if len(generators) != len(variables) or tuple(map(str, generators)) != variables:
        raise ValueError("SymPy generators do not match the declared order")
    result = Poly.from_dict(
        {
            term.exponents: Rational(*term.coefficient.as_integer_ratio())
            for term in polynomial.terms
        },
        *generators,
        domain=QQ,
    )
    if cache is not None:
        cache[key] = result
    return result


def sparse_rational_polynomial_from_sympy(
    polynomial: Any,
    variables: tuple[str, ...],
    *,
    maximum_terms: int = MAX_POLYNOMIAL_TERMS,
) -> SparseRationalPolynomial:
    """Convert a QQ ``Poly`` to canonical sparse term data."""

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
    return SparseRationalPolynomial(
        terms=tuple(
            RationalPolynomialTerm(
                coefficient=rational_from_sympy(coefficient),
                exponents=exponents,
            )
            for exponents, coefficient in terms
        )
    )


def rational_function_to_sympy(
    value: RationalFunction,
    *,
    symbols: tuple[Any, ...] | None = None,
    polynomial_cache: dict[tuple[tuple[str, ...], SparseRationalPolynomial], Any]
    | None = None,
    value_cache: dict[RationalFunction, Any] | None = None,
) -> Any:
    """Construct an exact SymPy expression from a canonical rational function.

    Optional dictionaries are request-local reuse points for matrix and tensor
    kernels that encounter the same canonical value more than once.
    """

    from sympy import Rational

    if value_cache is not None and value in value_cache:
        return value_cache[value]
    if not value.variables:
        result = (
            Rational(0)
            if not value.numerator.terms
            else Rational(*value.numerator.terms[0].coefficient.as_integer_ratio())
        )
    else:
        numerator = sparse_rational_polynomial_to_sympy(
            value.numerator,
            value.variables,
            symbols=symbols,
            cache=polynomial_cache,
        )
        denominator = sparse_rational_polynomial_to_sympy(
            value.denominator,
            value.variables,
            symbols=symbols,
            cache=polynomial_cache,
        )
        result = numerator.as_expr() / denominator.as_expr()
    if value_cache is not None:
        value_cache[value] = result
    return result


def rational_function_from_sympy(
    expression: Any,
    variables: tuple[str, ...],
    *,
    maximum_terms: int = 256,
    deadline_check: Any = None,
    symbols: tuple[Any, ...] | None = None,
) -> RationalFunction:
    """Canonicalize an exact rational-function expression into wire data.

    If ``deadline_check`` is provided, it is called before and after the
    potentially expensive ``cancel`` step so callers can enforce a request
    deadline around normalization.
    """

    from sympy import QQ, Poly, cancel, fraction

    if not variables:
        if isinstance(expression, bool):
            raise ValueError("boolean is not an exact rational")
        coefficient = (
            CanonicalRational.from_integer_ratio(expression, 1)
            if isinstance(expression, int)
            else rational_from_sympy(expression)
        )
        return RationalFunction._from_kernel(
            variables=(),
            numerator=SparseRationalPolynomial(
                terms=()
                if coefficient.as_fraction() == 0
                else (
                    RationalPolynomialTerm(
                        coefficient=coefficient,
                        exponents=(),
                    ),
                )
            ),
            denominator=SparseRationalPolynomial(
                terms=(
                    RationalPolynomialTerm(
                        coefficient=CanonicalRational(num=1, den=1),
                        exponents=(),
                    ),
                )
            ),
        )
    generators = symbols_for_variables(variables) if symbols is None else symbols
    if len(generators) != len(variables) or tuple(map(str, generators)) != variables:
        raise ValueError("SymPy generators do not match the declared order")
    if deadline_check is not None:
        deadline_check()
    numerator_expression, denominator_expression = fraction(cancel(expression))
    if deadline_check is not None:
        deadline_check()
    numerator = Poly(numerator_expression, *generators, domain=QQ)
    denominator = Poly(denominator_expression, *generators, domain=QQ)
    leading = denominator.LC()
    numerator = (
        numerator.monic() * (numerator.LC() / leading)
        if not numerator.is_zero
        else numerator
    )
    denominator = denominator.monic()
    return RationalFunction._from_kernel(
        variables=variables,
        numerator=sparse_rational_polynomial_from_sympy(
            numerator, variables, maximum_terms=maximum_terms
        ),
        denominator=sparse_rational_polynomial_from_sympy(
            denominator, variables, maximum_terms=maximum_terms
        ),
    )


def rational_polynomial_from_sympy(
    polynomial: Any,
    variables: tuple[str, ...],
    *,
    maximum_terms: int = MAX_POLYNOMIAL_TERMS,
) -> RationalPolynomial:
    """Convert a QQ ``Poly`` back to the canonical sparse contract."""

    return RationalPolynomial(
        variables=variables,
        polynomial=sparse_rational_polynomial_from_sympy(
            polynomial,
            variables,
            maximum_terms=maximum_terms,
        ),
    )
