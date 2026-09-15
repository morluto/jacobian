"""Property tests for the exact field discriminant.

The field discriminant is a signed invariant.  Its independent definition for a
monic defining polynomial ``f`` with root ``alpha`` is

    disc(f) = [O_K : ZZ[alpha]]**2 * disc(K),

so the polynomial discriminant must divide the field discriminant and the
quotient must be a positive perfect square.  A sign or scaling error cannot
satisfy that relation.
"""

from __future__ import annotations

from math import isqrt

import pytest
import sympy
from hypothesis import example, given, settings
from hypothesis import strategies as st

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.number_fields import (
    SimpleNumberFieldPresentation,
    discriminant,
)

_COEFFICIENTS = st.integers(min_value=-6, max_value=6)


@st.composite
def monic_defining_polynomials(draw) -> tuple[int, ...]:
    degree = draw(st.integers(min_value=2, max_value=3))
    return (1, *(draw(_COEFFICIENTS) for _ in range(degree)))


@settings(max_examples=15, deadline=None)
@given(coefficients=monic_defining_polynomials())
@example(coefficients=(1, 0, -5))
@example(coefficients=(1, 0, 0, -2))
def test_field_discriminant_is_polynomial_discriminant_over_square(
    coefficients: tuple[int, ...],
) -> None:
    polynomial = sympy.Poly.from_list(
        list(coefficients), sympy.Symbol("x"), domain=sympy.ZZ
    )
    if polynomial.is_irreducible is not True:
        return
    try:
        field_discriminant = discriminant(
            SimpleNumberFieldPresentation(coefficients_descending=coefficients)
        )
    except (OperationDomainValidationError, OperationResourceAdmissionError):
        return

    polynomial_discriminant = int(sympy.discriminant(polynomial))
    assert field_discriminant != 0
    assert polynomial_discriminant % field_discriminant == 0
    index_squared = polynomial_discriminant // field_discriminant
    assert index_squared > 0
    assert isqrt(index_squared) ** 2 == index_squared


@pytest.mark.parametrize("d", [2, 3, 5, 6, 7, 10, 13, -1, -2, -3, -5, -6, -7])
def test_quadratic_field_discriminant_matches_the_known_formula(d: int) -> None:
    """For squarefree ``d``, ``disc(QQ(sqrt(d)))`` is ``d`` or ``4d`` by parity."""

    field = SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -d))
    assert discriminant(field) == (d if d % 4 == 1 else 4 * d)
