"""Native domain functions accepting canonical polynomial values."""

from __future__ import annotations

from jacobian._exact import CanonicalRational
from jacobian.math.polynomial_support_geometry.operations import (
    _compute_weight_layers,
    _initial_form_terms,
    newton_polytope_from_polynomial,
    support_from_polynomial,
)
from jacobian.math.polynomial_support_geometry.values import (
    NewtonPolytope,
    PolynomialFaceData,
    PolynomialSupport,
    PolynomialWeightProfile,
)
from jacobian.math.polynomials.values import (
    RationalPolynomial,
    RationalPolynomialTerm,
    SparseRationalPolynomial,
)

__all__ = [
    "exponent_support",
    "initial_form",
    "newton_polytope",
    "weight_profile",
]


def exponent_support(polynomial: RationalPolynomial) -> PolynomialSupport:
    """Exponent support of one canonical polynomial value."""
    return support_from_polynomial(polynomial)


def newton_polytope(polynomial: RationalPolynomial) -> NewtonPolytope:
    """Newton polytope of one canonical polynomial value."""
    return newton_polytope_from_polynomial(polynomial)


def weight_profile(
    polynomial: RationalPolynomial, weight: tuple[int, ...]
) -> PolynomialWeightProfile:
    """Weight profile of a canonical polynomial under an integer weight.

    Native callers bypass the wire transport caps and use the shared
    kernel directly; the value validator still replays the profile.
    """
    minimum_weight, minimizing, layers = _compute_weight_layers(polynomial, weight)
    return PolynomialWeightProfile(
        polynomial=polynomial,
        weight=weight,
        minimum_weight=minimum_weight,
        minimizing_exponents=minimizing,
        weight_layers=layers,
    )


def initial_form(
    polynomial: RationalPolynomial, weight: tuple[int, ...]
) -> PolynomialFaceData:
    """Initial form of a canonical polynomial under an integer weight.

    Native callers bypass the wire transport caps and build the face via
    the shared kernel; the value validator still replays it.
    """
    face_terms = _initial_form_terms(polynomial, weight)
    initial_form_value = RationalPolynomial(
        variables=polynomial.variables,
        polynomial=SparseRationalPolynomial(
            terms=tuple(
                RationalPolynomialTerm(
                    coefficient=CanonicalRational.from_fraction(c), exponents=e
                )
                for c, e in face_terms
            )
        ),
    )
    return PolynomialFaceData(
        polynomial=polynomial,
        weight=weight,
        initial_form=initial_form_value,
    )
