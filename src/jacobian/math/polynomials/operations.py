"""Exact polynomial operations on canonical SymPy ``Poly`` inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jacobian.math.polynomials.values import RationalFunction

if TYPE_CHECKING:
    from sympy import Poly

__all__ = [
    "derivative",
    "discriminant",
    "divide",
    "evaluate",
    "factorization",
    "gcdex",
    "groebner_basis",
    "hermite_reduction",
    "integral",
    "partial_fractions",
    "resultant",
    "square_free_decomposition",
]


def _poly(value: Poly) -> Poly:
    from sympy import Poly

    if not isinstance(value, Poly):
        raise TypeError("polynomial must be a SymPy Poly")
    return value


def gcdex(left: Poly, right: Poly) -> tuple[Poly, Poly, Poly]:
    """Return the exact extended-GCD tuple for two compatible polynomials."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_gcdex(_poly(left), _poly(right))


def resultant(left: Poly, right: Poly, generator: Any) -> Any:
    """Return the exact resultant in the supplied common generator."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_resultant(_poly(left), _poly(right), generator)


def derivative(polynomial: Poly) -> Poly:
    """Return the formal derivative of a polynomial."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_derivative(_poly(polynomial))


def discriminant(polynomial: Poly, generator: Any) -> Any:
    """Return the discriminant in the supplied generator."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_discriminant(_poly(polynomial), generator)


def divide(left: Poly, right: Poly) -> tuple[Poly, Poly, Poly]:
    """Return quotient, remainder, and exact reconstruction."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_division(_poly(left), _poly(right))


def evaluate(polynomial: Poly, point: Any) -> Any:
    """Evaluate a polynomial at one exact backend-native point."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_evaluate(_poly(polynomial), point)


def factorization(source: Poly) -> tuple[Any, tuple[tuple[Poly, int], ...], Poly]:
    """Return coefficient, monic irreducible factors, and reconstruction."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_factorization(_poly(source))


def groebner_basis(
    generators: tuple[Poly, ...],
    variables: tuple[Any, ...],
    monomial_order: str,
) -> tuple[Poly, ...]:
    """Return a reduced Gröbner basis over ``QQ``."""

    from jacobian.math.polynomials import _sympy

    canonical_generators = tuple(_poly(generator) for generator in generators)
    if any(not generator.domain.is_QQ for generator in canonical_generators):
        raise ValueError("Gröbner basis generators must use the QQ domain")
    return _sympy.polynomial_groebner_basis(
        canonical_generators,
        variables,
        monomial_order,
    )


def integral(polynomial: Poly) -> Poly:
    """Return the formal antiderivative with zero constant term."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_integral(_poly(polynomial))


def hermite_reduction(
    function: RationalFunction,
) -> tuple[RationalFunction, RationalFunction]:
    """Reduce one admitted canonical rational function modulo derivatives."""

    from jacobian.math.polynomials.rational_functions.operations import (
        hermite_reduction as _hermite_reduction,
    )

    return _hermite_reduction(function)


def partial_fractions(expression: Any, generator: Any) -> Any:
    """Return an exact univariate partial-fraction decomposition."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_partial_fractions(expression, generator)


def square_free_decomposition(
    source: Poly,
) -> tuple[Any, tuple[tuple[Poly, int], ...], Poly]:
    """Return coefficient, monic square-free factors, and reconstruction."""

    from jacobian.math.polynomials import _sympy

    return _sympy.polynomial_square_free_decomposition(_poly(source))
