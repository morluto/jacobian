"""Private SymPy arithmetic for the always-available finite-field surface."""

from __future__ import annotations

from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
    FinitePolynomial,
)


def evaluate_polynomial_values(
    polynomial: FinitePolynomial,
    values: tuple[FiniteFieldElement, ...],
) -> tuple[tuple[int, ...], ...]:
    """Evaluate several points with one prepared exact quotient presentation."""

    from sympy import Poly, symbols

    variable = symbols("z")
    characteristic = polynomial.presentation.characteristic
    modulus = Poly(
        sum(
            coefficient * variable**power
            for power, coefficient in enumerate(
                polynomial.presentation.modulus_coefficients
            )
        ),
        variable,
        modulus=characteristic,
    )

    def as_polynomial(element: FiniteFieldElement) -> Poly:
        return Poly(
            sum(
                coefficient * variable**power
                for power, coefficient in enumerate(element.coordinates)
            ),
            variable,
            modulus=characteristic,
        )

    coefficients = tuple(as_polynomial(value) for value in polynomial.coefficients)

    def evaluate(value: FiniteFieldElement) -> tuple[int, ...]:
        point = as_polynomial(value)
        evaluated = Poly(0, variable, modulus=characteristic)
        for coefficient in reversed(coefficients):
            evaluated = (evaluated * point + coefficient).rem(modulus)
        return tuple(
            int(evaluated.nth(power)) % characteristic
            for power in range(polynomial.presentation.degree)
        )

    return tuple(evaluate(value) for value in values)


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
