"""Native domain functions accepting canonical polynomial values."""

from __future__ import annotations

from jacobian.math.polynomial_support_geometry._models import (
    InitialFormRequest,
    NewtonPolytopeRequest,
    SupportRequest,
    WeightProfileRequest,
)
from jacobian.math.polynomial_support_geometry.values import (
    NewtonPolytope,
    PolynomialFaceData,
    PolynomialSupport,
    PolynomialWeightProfile,
)
from jacobian.math.polynomials.values import RationalPolynomial

__all__ = [
    "exponent_support",
    "initial_form",
    "newton_polytope",
    "weight_profile",
]


def exponent_support(
    polynomial: RationalPolynomial,
) -> PolynomialSupport:
    """Exponent support of one canonical polynomial value."""
    return compute_support(SupportRequest(polynomial=polynomial))


def newton_polytope(
    polynomial: RationalPolynomial,
) -> NewtonPolytope:
    """Newton polytope of one canonical polynomial value."""
    return compute_newton_polytope(NewtonPolytopeRequest(polynomial=polynomial))


def weight_profile(
    polynomial: RationalPolynomial, weight: tuple[int, ...]
) -> PolynomialWeightProfile:
    """Weight profile of a canonical polynomial under an integer weight."""
    return compute_weight_profile(
        WeightProfileRequest(polynomial=polynomial, weight=weight)
    )


def initial_form(
    polynomial: RationalPolynomial, weight: tuple[int, ...]
) -> PolynomialFaceData:
    """Initial form of a canonical polynomial under an integer weight."""
    return compute_initial_form(
        InitialFormRequest(polynomial=polynomial, weight=weight)
    )


from jacobian.math.polynomial_support_geometry.operations import (  # noqa: E402
    compute_initial_form,
    compute_newton_polytope,
    compute_support,
    compute_weight_profile,
)
