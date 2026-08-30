"""Private SymPy kernel for canonical simple-field presentations."""

from __future__ import annotations

from typing import Any, cast

from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldPresentation,
)


def recognized_integral_basis(
    field: SimpleNumberFieldPresentation,
) -> tuple[Any, Any, Any, int] | None:
    """Recognize the presentation and compute its integral basis once."""

    import sympy
    from sympy.polys.numberfields import round_two

    alpha = sympy.Symbol("alpha")
    coefficients = tuple(
        parse_canonical_integer(coefficient)
        for coefficient in field.coefficients_descending
    )
    leading = coefficients[0]
    # beta = leading * alpha has the monic integral polynomial
    # leading^(n-1) f(beta / leading). This preserves QQ(alpha) while
    # allowing the canonical presentation itself to remain nonmonic.
    monic_coefficients = (
        1,
        *(
            coefficient * leading ** (index - 1)
            for index, coefficient in enumerate(coefficients[1:], start=1)
        ),
    )
    polynomial = sympy.Poly.from_list(
        monic_coefficients,
        gens=alpha,
        domain=sympy.ZZ,
    )
    if polynomial.is_irreducible is not True:
        return None
    ring, field_discriminant = cast(tuple[Any, Any], round_two(polynomial))
    return ring, field_discriminant, alpha, leading


__all__ = [
    "recognized_integral_basis",
]
