"""Exact polynomial support geometry over canonical polynomial values."""

from jacobian.math.polynomial_support_geometry.native import (
    exponent_support,
    initial_form,
    newton_polytope,
    weight_profile,
)

# Wire-envelope handlers and value types used by the MCP projection.
from jacobian.math.polynomial_support_geometry.operations import (
    compute_initial_form,
    compute_newton_polytope,
    compute_support,
    compute_weight_profile,
)
from jacobian.math.polynomial_support_geometry.values import (
    NewtonPolytope,
    PolynomialFaceData,
    PolynomialSupport,
    PolynomialWeightProfile,
)

# The first four names are the native surface: they accept canonical
# domain values directly. The remainder are wire-envelope handlers and
# value types used by the MCP projection.
__all__ = [
    "NewtonPolytope",
    "PolynomialFaceData",
    "PolynomialSupport",
    "PolynomialWeightProfile",
    "compute_initial_form",
    "compute_newton_polytope",
    "compute_support",
    "compute_weight_profile",
    "exponent_support",
    "initial_form",
    "newton_polytope",
    "weight_profile",
]
