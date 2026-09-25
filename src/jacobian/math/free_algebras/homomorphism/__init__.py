"""Exact homomorphisms between finitely generated free associative algebras."""

from jacobian.math.free_algebras.homomorphism._models import (
    FreeAlgebraHomomorphismApplyRequest,
)
from jacobian.math.free_algebras.homomorphism.operations import apply

__all__ = [
    "FreeAlgebraHomomorphismApplyRequest",
    "apply",
]
