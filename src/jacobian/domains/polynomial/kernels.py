"""Thin typed calls into SymPy's exact polynomial algorithms."""

from __future__ import annotations

from typing import Any, cast

__all__ = [
    "polynomial_derivative",
    "polynomial_discriminant",
    "polynomial_division",
    "polynomial_evaluate",
    "polynomial_gcdex",
    "polynomial_groebner_basis",
    "polynomial_integral",
    "polynomial_partial_fractions",
    "polynomial_resultant",
    "polynomial_square_free_decomposition",
]


def polynomial_gcdex(left: Any, right: Any) -> tuple[Any, Any, Any]:
    return cast(tuple[Any, Any, Any], left.gcdex(right))


def polynomial_resultant(left: Any, right: Any, generator: Any) -> Any:
    from sympy import resultant

    return resultant(left.as_expr(), right.as_expr(), generator)


def polynomial_discriminant(polynomial: Any, generator: Any) -> Any:
    from sympy import discriminant

    return discriminant(polynomial.as_expr(), generator)


def polynomial_square_free_decomposition(
    source: Any,
) -> tuple[Any, tuple[tuple[Any, int], ...], Any]:
    """Return coefficient, monic factors, and the exact reconstruction."""

    from sympy import Poly

    coefficient, raw_factors = source.sqf_list()
    factors = tuple(
        (factor.monic(), int(multiplicity)) for factor, multiplicity in raw_factors
    )
    reconstructed = Poly(coefficient, *source.gens, domain=source.domain)
    for factor, multiplicity in factors:
        reconstructed *= factor**multiplicity
    if reconstructed != source:
        raise RuntimeError("SymPy square-free decomposition did not reconstruct input")
    return coefficient, factors, reconstructed


def polynomial_groebner_basis(
    generators: tuple[Any, ...],
    variables: tuple[Any, ...],
    monomial_order: str,
) -> tuple[Any, ...]:
    from sympy import QQ, groebner

    basis = groebner(
        [generator.as_expr() for generator in generators],
        *variables,
        order=monomial_order,
        domain=QQ,
    )
    return tuple(basis.polys)


def polynomial_division(left: Any, right: Any) -> tuple[Any, Any, Any]:
    quotient, remainder = left.div(right)
    return quotient, remainder, quotient * right + remainder


def polynomial_evaluate(polynomial: Any, point: Any) -> Any:
    return polynomial.eval(point)


def polynomial_derivative(polynomial: Any) -> Any:
    return polynomial.diff()


def polynomial_integral(polynomial: Any) -> Any:
    return polynomial.integrate()


def polynomial_partial_fractions(expression: Any, generator: Any) -> Any:
    from sympy import apart

    return apart(expression, generator)
