"""Exact tropical semiring scalars and idempotent addition."""

from jacobian.math.polynomials.tropical._models import ScalarAddRequest, ScalarAddResult
from jacobian.math.polynomials.tropical.operations import tropical_scalar_add
from jacobian.math.polynomials.tropical.values import (
    TropicalScalar,
    TropicalSemiring,
    require_scalar_budget,
)

__all__ = [
    "ScalarAddRequest",
    "ScalarAddResult",
    "TropicalScalar",
    "TropicalSemiring",
    "require_scalar_budget",
    "tropical_scalar_add",
]
