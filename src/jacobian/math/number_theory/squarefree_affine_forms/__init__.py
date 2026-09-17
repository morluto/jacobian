"""Canonical values and native operations for square-free affine forms."""

from jacobian.math.number_theory.squarefree_affine_forms._admissibility import (
    LocalAdmissibilityRequest,
    LocalAdmissibilityResult,
)
from jacobian.math.number_theory.squarefree_affine_forms._interval_count import (
    IntervalCountRequest,
    IntervalCountResult,
    IntervalObstruction,
)
from jacobian.math.number_theory.squarefree_affine_forms.operations import (
    euler_product,
    interval_count,
    local_admissibility,
    local_factor,
    verify_interval_count,
    verify_local_admissibility,
    verify_squarefree_affine_family,
)
from jacobian.math.number_theory.squarefree_affine_forms.values import (
    SquarefreeAffineFamily,
    SquarefreeAffineForm,
)

__all__ = [
    "IntervalCountRequest",
    "IntervalCountResult",
    "IntervalObstruction",
    "LocalAdmissibilityRequest",
    "LocalAdmissibilityResult",
    "SquarefreeAffineFamily",
    "SquarefreeAffineForm",
    "euler_product",
    "interval_count",
    "local_admissibility",
    "local_factor",
    "verify_interval_count",
    "verify_local_admissibility",
    "verify_squarefree_affine_family",
]
