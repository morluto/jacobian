"""Domain-owned prime-field matrix operations."""

from __future__ import annotations

from jacobian.math.prime_field_linear_algebra import (
    nullspace as _nullspace,
)
from jacobian.math.prime_field_linear_algebra import (
    rank as _rank,
)
from jacobian.math.prime_field_linear_algebra import (
    rref as _rref,
)
from jacobian.math.prime_field_matrix._models import (
    PrimeFieldMatrixRankResult,
    PrimeFieldMatrixRequest,
    PrimeFieldNullspaceResult,
    PrimeFieldRrefResult,
    _kernel_matrix,
)


def compute_rank(request: PrimeFieldMatrixRequest) -> PrimeFieldMatrixRankResult:
    """Compute the rank of a matrix over GF(p)."""
    matrix = _kernel_matrix(request)
    return PrimeFieldMatrixRankResult(
        source=request,
        prime=matrix.prime,
        rank=_rank(matrix),
    )


def compute_rref(request: PrimeFieldMatrixRequest) -> PrimeFieldRrefResult:
    """Compute the reduced row-echelon form of a matrix over GF(p)."""
    matrix = _kernel_matrix(request)
    rref_rows, pivot_columns = _rref(matrix)
    return PrimeFieldRrefResult(
        source=request,
        prime=matrix.prime,
        rref=rref_rows,
        pivot_columns=pivot_columns,
        rank=len(pivot_columns),
    )


def compute_nullspace(request: PrimeFieldMatrixRequest) -> PrimeFieldNullspaceResult:
    """Compute a basis for the right nullspace of a matrix over GF(p)."""
    matrix = _kernel_matrix(request)
    basis = _nullspace(matrix)
    return PrimeFieldNullspaceResult(
        source=request,
        prime=matrix.prime,
        nullspace=basis,
        nullity=len(basis),
    )


__all__ = [
    "compute_nullspace",
    "compute_rank",
    "compute_rref",
]
