"""Private SymPy arithmetic for the always-available finite-field surface."""

from __future__ import annotations

from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
)


def normalize_projective_coordinates(
    presentation: FiniteFieldPresentation,
    coordinates: tuple[FiniteFieldElement, ...],
) -> tuple[tuple[int, ...], ...]:
    """Normalize homogeneous coordinates with SymPy quotient arithmetic."""

    from sympy import Poly, invert, symbols

    variable = symbols("z")
    modulus = Poly(
        sum(
            coefficient * variable**power
            for power, coefficient in enumerate(presentation.modulus_coefficients)
        ),
        variable,
        modulus=presentation.characteristic,
    )

    def polynomial(value: FiniteFieldElement) -> Poly:
        return Poly(
            sum(
                coefficient * variable**power
                for power, coefficient in enumerate(value.coordinates)
            ),
            variable,
            modulus=presentation.characteristic,
        )

    values = tuple(polynomial(value) for value in coordinates)
    pivot = next((value for value in values if not value.is_zero), None)
    if pivot is None:
        raise ValueError("projective coordinates cannot all be zero")
    inverse = invert(pivot, modulus)
    normalized = tuple((value * inverse).rem(modulus) for value in values)
    return tuple(
        tuple(
            int(value.nth(power)) % presentation.characteristic
            for power in range(presentation.degree)
        )
        for value in normalized
    )
