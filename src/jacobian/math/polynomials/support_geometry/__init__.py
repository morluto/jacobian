"""Exact polynomial support geometry over canonical polynomial values."""

from jacobian.math.polynomials.support_geometry.operations import (
    exponent_support,
    initial_form,
    newton_polytope,
    weight_profile,
)
from jacobian.math.polynomials.support_geometry.values import (
    NewtonPolytope,
    PolynomialFaceData,
    PolynomialSupport,
    PolynomialWeightProfile,
)

# The native surface accepts canonical domain values directly. The
# compute_* wire-envelope handlers stay private to the MCP declarations
# (they require operation-specific request envelopes) and are not part
# of the supported Python API.
__all__ = [
    "NewtonPolytope",
    "PolynomialFaceData",
    "PolynomialSupport",
    "PolynomialWeightProfile",
    "exponent_support",
    "initial_form",
    "newton_polytope",
    "weight_profile",
]
