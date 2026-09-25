"""Exact scalar action on sparse free-algebra polynomials."""

from jacobian.math.free_algebras.scalar_multiply._models import (
    FreeAlgebraPolynomialScalarMultiplyRequest,
)
from jacobian.math.free_algebras.scalar_multiply.operations import scalar_multiply

__all__ = ["FreeAlgebraPolynomialScalarMultiplyRequest", "scalar_multiply"]
