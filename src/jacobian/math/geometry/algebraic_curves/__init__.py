"""Plane algebraic curve operations."""

from jacobian.math.geometry.algebraic_curves._singularity_models import (
    ProjectivePlaneCurveSingularityBudget,
    ProjectivePlaneCurveSingularityProfile,
)
from jacobian.math.geometry.algebraic_curves.operations import (
    affine_chart,
    affine_curve_check,
    projective_closure,
    rational_conic_parametrization,
    singularity_profile,
)

__all__ = [
    "ProjectivePlaneCurveSingularityBudget",
    "ProjectivePlaneCurveSingularityProfile",
    "affine_chart",
    "affine_curve_check",
    "projective_closure",
    "rational_conic_parametrization",
    "singularity_profile",
]
