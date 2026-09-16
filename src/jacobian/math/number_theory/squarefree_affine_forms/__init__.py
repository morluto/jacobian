"""Canonical values and native operations for square-free affine forms."""

from jacobian.math.number_theory.squarefree_affine_forms.operations import (
    euler_product,
    local_factor,
    verify_squarefree_affine_family,
)
from jacobian.math.number_theory.squarefree_affine_forms.values import (
    SquarefreeAffineFamily,
    SquarefreeAffineForm,
)

__all__ = [
    "SquarefreeAffineFamily",
    "SquarefreeAffineForm",
    "euler_product",
    "local_factor",
    "verify_squarefree_affine_family",
]
