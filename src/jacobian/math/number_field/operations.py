"""Number field operations backed by SymPy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

__all__ = ["discriminant", "ring_of_integers"]


def _integral_basis(
    coefficients_descending: Sequence[str], variable: str
) -> tuple[Any, Any]:
    import sympy
    from sympy.polys.numberfields import round_two

    x = sympy.Symbol(variable)
    polynomial = sum(
        sympy.Rational(coefficient) * x ** (len(coefficients_descending) - 1 - index)
        for index, coefficient in enumerate(coefficients_descending)
    )
    return cast(tuple[Any, Any], round_two(sympy.Poly(polynomial, x)))


def discriminant(coefficients_descending: Sequence[str], variable: str) -> str:
    _ring_of_integers, field_discriminant = _integral_basis(
        coefficients_descending, variable
    )
    return str(field_discriminant)


def ring_of_integers(
    coefficients_descending: Sequence[str], variable: str
) -> list[str]:
    """Return the exact integral basis expressed in the defining power basis."""
    ring, _field_discriminant = _integral_basis(coefficients_descending, variable)
    return [str(element.as_expr()) for element in ring.basis_element_pullbacks()]
