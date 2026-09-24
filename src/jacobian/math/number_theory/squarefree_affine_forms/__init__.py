"""Canonical values and native operations for square-free affine forms."""

from jacobian.math.number_theory.squarefree_affine_forms._admissibility import (
    LocalAdmissibilityResult,
)
from jacobian.math.number_theory.squarefree_affine_forms._infinite_product import (
    SquarefreeInfiniteProductEnclosure,
)
from jacobian.math.number_theory.squarefree_affine_forms._interval_count import (
    IntervalCountResult,
    IntervalObstruction,
)
from jacobian.math.number_theory.squarefree_affine_forms.operations import (
    euler_product,
    infinite_product_enclosure,
    interval_count,
    local_admissibility,
    local_factor,
    verify_squarefree_affine_family,
)
from jacobian.math.number_theory.squarefree_affine_forms.values import (
    SquarefreeAffineFamily,
    SquarefreeAffineForm,
)

__all__ = [
    "IntervalCountResult",
    "IntervalObstruction",
    "LocalAdmissibilityResult",
    "SquarefreeAffineFamily",
    "SquarefreeAffineForm",
    "SquarefreeInfiniteProductEnclosure",
    "euler_product",
    "infinite_product_enclosure",
    "interval_count",
    "local_admissibility",
    "local_factor",
    "verify_squarefree_affine_family",
]
