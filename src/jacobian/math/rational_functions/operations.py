"""Exact native operations on rational functions."""

from __future__ import annotations

from typing import Any

from jacobian.math.polynomials._conversions import (
    rational_function_from_sympy,
    rational_function_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import RationalFunction


def _hermite_parts(function: RationalFunction) -> tuple[Any, Any]:
    """Return the zero-constant rational part and square-free remainder."""

    from sympy import Poly, cancel, diff, fraction
    from sympy.integrals.rationaltools import ratint_ratpart

    (variable,) = symbols_for_variables(function.variables)
    source = cancel(rational_function_to_sympy(function))
    numerator_expression, denominator_expression = fraction(source)
    numerator = Poly(numerator_expression, variable, domain="QQ")
    denominator = Poly(denominator_expression, variable, domain="QQ")
    polynomial_part, proper_numerator = numerator.div(denominator)
    rational_part = polynomial_part.integrate().as_expr()

    if proper_numerator.is_zero:
        remainder = 0
    else:
        repeated_pole_part, remainder = ratint_ratpart(
            proper_numerator,
            denominator,
            variable,
        )
        rational_part += repeated_pole_part

    rational_part = cancel(rational_part)
    remainder = cancel(remainder)
    if cancel(diff(rational_part, variable) + remainder - source) != 0:
        raise AssertionError("Hermite reduction failed exact reconstruction")
    return rational_part, remainder


def hermite_reduction(
    function: RationalFunction,
) -> tuple[RationalFunction, RationalFunction]:
    """Compute canonical ``f = R' + H`` over ``QQ(x)``.

    ``H`` is proper with square-free denominator. ``R`` uses zero as its
    additive constant, so the pair is unique for the source function.
    """

    if len(function.variables) != 1:
        raise ValueError("Hermite reduction requires exactly one variable")
    rational_part, remainder = _hermite_parts(function)
    return (
        rational_function_from_sympy(rational_part, function.variables),
        rational_function_from_sympy(remainder, function.variables),
    )


__all__ = ["hermite_reduction"]
