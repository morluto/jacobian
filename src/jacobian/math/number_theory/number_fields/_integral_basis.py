"""Private SymPy kernel for canonical simple-field presentations."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, cast

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.math.number_theory.number_fields.values import (
    MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
    SimpleNumberFieldPresentation,
)


def recognized_integral_basis(
    field: SimpleNumberFieldPresentation,
) -> tuple[Any, Any, Any, int] | None:
    """Recognize the presentation and compute its integral basis once."""

    import sympy
    from sympy.polys.numberfields import round_two

    alpha = sympy.Symbol("alpha")
    coefficients = tuple(coefficient for coefficient in field.coefficients_descending)
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


def integral_basis_coordinates(
    field: SimpleNumberFieldPresentation,
    recognized: tuple[Any, Any, Any, int] | None = None,
) -> tuple[tuple[CanonicalRational, ...], ...] | None:
    """Return the recognized basis on the presented generator's power basis."""

    if recognized is None:
        recognized = recognized_integral_basis(field)
    if recognized is None:
        return None
    ring, _field_discriminant, alpha, leading = recognized
    basis: list[tuple[CanonicalRational, ...]] = []
    for element in ring.basis_element_pullbacks():
        expression = element.as_expr().subs(alpha, leading * alpha).expand()
        polynomial = _as_poly_in_alpha(expression, alpha)
        coefficients = polynomial.all_coeffs()[::-1]
        padded = [_rational(coefficient) for coefficient in coefficients]
        padded.extend(
            CanonicalRational(num=0, den=1) for _ in range(field.degree - len(padded))
        )
        if len(padded) != field.degree:
            raise ValueError(
                "an integral basis vector must span the complete power basis"
            )
        for coefficient in padded:
            require_bounded_rational(
                coefficient,
                max_digits=MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
                label="integral basis",
            )
        basis.append(tuple(padded))
    return tuple(basis)


def _rational(value: Any) -> CanonicalRational:
    fraction = Fraction(value)
    return CanonicalRational(num=fraction.numerator, den=fraction.denominator)


def _as_poly_in_alpha(expression: Any, alpha: Any) -> Any:
    import sympy

    return sympy.Poly(expression, alpha)


__all__ = [
    "integral_basis_coordinates",
    "recognized_integral_basis",
]
