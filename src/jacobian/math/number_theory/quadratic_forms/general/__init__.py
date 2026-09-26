"""Exact rational quadratic-form values and direct evaluation."""

from jacobian.math.number_theory.quadratic_forms.general._extra_models import (
    FiniteBoxProfileResult,
    FiniteGaussSumResult,
    ThetaRepresentingVectorsRequest,
    ThetaRepresentingVectorsResult,
    ThetaSelectedCoefficient,
    ThetaSelectedCoefficientsResult,
    ThetaSeriesPrefixResult,
)
from jacobian.math.number_theory.quadratic_forms.general.direct_sum_models import (
    QuadraticFormDirectSumResult,
    QuadraticFormRestrictionResult,
)
from jacobian.math.number_theory.quadratic_forms.general.direct_sum_operations import (
    quadratic_form_direct_sum,
    quadratic_form_restrict_coordinates,
)
from jacobian.math.number_theory.quadratic_forms.general.extra_operations import (
    finite_quadratic_gauss_sum,
)
from jacobian.math.number_theory.quadratic_forms.general.finite_box_operations import (
    finite_box_value_profile,
)
from jacobian.math.number_theory.quadratic_forms.general.operations import (
    bilinear_pairing,
    coefficient_matrix,
    coefficient_matrix_entries,
    evaluate_rational_quadratic_form,
    require_coefficient_matrix_budget,
)
from jacobian.math.number_theory.quadratic_forms.general.scaling_models import (
    QuadraticFormScaleRequest,
    QuadraticFormScaleResult,
)
from jacobian.math.number_theory.quadratic_forms.general.scaling_operations import (
    scale_rational_quadratic_form,
)
from jacobian.math.number_theory.quadratic_forms.general.theta_operations import (
    theta_representing_vectors,
    theta_selected_coefficients,
    theta_series_prefix,
)
from jacobian.math.number_theory.quadratic_forms.general.values import (
    QuadraticCrossTerm,
    RationalCoordinateVector,
    RationalQuadraticForm,
)

__all__ = [
    "FiniteBoxProfileResult",
    "FiniteGaussSumResult",
    "QuadraticCrossTerm",
    "QuadraticFormDirectSumResult",
    "QuadraticFormRestrictionResult",
    "QuadraticFormScaleRequest",
    "QuadraticFormScaleResult",
    "RationalCoordinateVector",
    "RationalQuadraticForm",
    "ThetaRepresentingVectorsRequest",
    "ThetaRepresentingVectorsResult",
    "ThetaSelectedCoefficient",
    "ThetaSelectedCoefficientsResult",
    "ThetaSeriesPrefixResult",
    "bilinear_pairing",
    "coefficient_matrix",
    "coefficient_matrix_entries",
    "evaluate_rational_quadratic_form",
    "finite_box_value_profile",
    "finite_quadratic_gauss_sum",
    "quadratic_form_direct_sum",
    "quadratic_form_restrict_coordinates",
    "require_coefficient_matrix_budget",
    "scale_rational_quadratic_form",
    "theta_representing_vectors",
    "theta_selected_coefficients",
    "theta_series_prefix",
]
