"""Plane algebraic curve operations."""

from jacobian.math.geometry.algebraic_curves._arclength_models import (
    PlaneCurveArclengthBudget,
    PlaneCurveArclengthResult,
)
from jacobian.math.geometry.algebraic_curves._gaussian_realification import (
    GaussianRealificationResult,
    UnivariateGaussianPolynomial,
    UnivariateGaussianPolynomialTerm,
)
from jacobian.math.geometry.algebraic_curves._models import (
    PlaneCurveBlowupChartResult,
)
from jacobian.math.geometry.algebraic_curves._singularity_models import (
    ProjectivePlaneCurveSingularityBudget,
    ProjectivePlaneCurveSingularityProfile,
)
from jacobian.math.geometry.algebraic_curves.operations import (
    affine_chart,
    affine_curve_check,
    enclose_arclength,
    gaussian_realification,
    plane_curve_blowup_chart,
    projective_closure,
    rational_conic_parametrization,
    singularity_profile,
    verify_affine_chart,
    verify_affine_curve_check,
    verify_gaussian_realification,
    verify_projective_closure,
    verify_projective_plane_curve_singularity_profile,
    verify_rational_conic_parametrization,
)

__all__ = [
    "GaussianRealificationResult",
    "PlaneCurveArclengthBudget",
    "PlaneCurveArclengthResult",
    "PlaneCurveBlowupChartResult",
    "ProjectivePlaneCurveSingularityBudget",
    "ProjectivePlaneCurveSingularityProfile",
    "UnivariateGaussianPolynomial",
    "UnivariateGaussianPolynomialTerm",
    "affine_chart",
    "affine_curve_check",
    "enclose_arclength",
    "gaussian_realification",
    "plane_curve_blowup_chart",
    "projective_closure",
    "rational_conic_parametrization",
    "singularity_profile",
    "verify_affine_chart",
    "verify_affine_curve_check",
    "verify_gaussian_realification",
    "verify_projective_closure",
    "verify_projective_plane_curve_singularity_profile",
    "verify_rational_conic_parametrization",
]
