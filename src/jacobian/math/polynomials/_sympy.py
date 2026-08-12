"""Private SymPy backend for exact polynomial algorithms."""

from __future__ import annotations

from typing import Any, cast


def polynomial_gcdex(left: Any, right: Any) -> tuple[Any, Any, Any]:
    return cast(tuple[Any, Any, Any], left.gcdex(right))


def polynomial_resultant(left: Any, right: Any, generator: Any) -> Any:
    from sympy import resultant

    return resultant(left.as_expr(), right.as_expr(), generator)


def polynomial_discriminant(polynomial: Any, generator: Any) -> Any:
    from sympy import discriminant

    return discriminant(polynomial, generator)


def _monic_decomposition(
    source: Any,
    decomposition: tuple[Any, list[tuple[Any, int]]],
    *,
    label: str,
) -> tuple[Any, tuple[tuple[Any, int], ...], Any]:
    from sympy import Poly

    coefficient, raw_factors = decomposition
    factors = []
    for factor, multiplicity in raw_factors:
        coefficient *= factor.LC() ** int(multiplicity)
        factors.append((factor.monic(), int(multiplicity)))
    reconstructed = Poly(coefficient, *source.gens, domain=source.domain)
    for factor, multiplicity in factors:
        reconstructed *= factor**multiplicity
    reconstructed = Poly(
        reconstructed.as_expr(),
        *source.gens,
        domain=source.domain,
    )
    if reconstructed != source:
        raise RuntimeError(f"SymPy {label} did not reconstruct input")
    return coefficient, tuple(factors), reconstructed


def polynomial_square_free_decomposition(
    source: Any,
) -> tuple[Any, tuple[tuple[Any, int], ...], Any]:
    return _monic_decomposition(
        source,
        source.sqf_list(),
        label="square-free decomposition",
    )


def polynomial_factorization(
    source: Any,
) -> tuple[Any, tuple[tuple[Any, int], ...], Any]:
    return _monic_decomposition(
        source,
        source.factor_list(),
        label="factorization",
    )


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
