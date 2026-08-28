"""Domain-owned prime-field matrix operations."""

from __future__ import annotations

from jacobian.math.matrices.finite_fields._models import (
    PrimeFieldMatrixRankResult,
    PrimeFieldMatrixRequest,
    PrimeFieldNullspaceResult,
    PrimeFieldRrefResult,
)
from jacobian.math.matrices.finite_fields.linear_algebra import (
    PrimeFieldMatrix,
)
from jacobian.math.matrices.finite_fields.linear_algebra import (
    nullspace as _nullspace,
)
from jacobian.math.matrices.finite_fields.linear_algebra import (
    rank as _rank,
)
from jacobian.math.matrices.finite_fields.linear_algebra import (
    rref as _rref,
)


def compute_rank(request: PrimeFieldMatrixRequest) -> PrimeFieldMatrixRankResult:
    """Compute the rank of a matrix over GF(p)."""
    return PrimeFieldMatrixRankResult._from_kernel(request, rank=_rank(request.matrix))


def compute_rref(request: PrimeFieldMatrixRequest) -> PrimeFieldRrefResult:
    """Compute the reduced row-echelon form of a matrix over GF(p)."""
    rref_rows, pivot_columns = _rref(request.matrix)
    return PrimeFieldRrefResult._from_kernel(
        request,
        rref_matrix=PrimeFieldMatrix(
            prime=request.matrix.prime,
            entries=tuple(rref_rows),
            columns=request.matrix.columns,
        ),
        pivot_columns=pivot_columns,
    )


def compute_nullspace(request: PrimeFieldMatrixRequest) -> PrimeFieldNullspaceResult:
    """Compute a basis for the right nullspace of a matrix over GF(p)."""
    basis = _nullspace(request.matrix)
    return PrimeFieldNullspaceResult._from_kernel(
        request,
        nullspace_matrix=PrimeFieldMatrix(
            prime=request.matrix.prime,
            entries=tuple(basis),
            columns=request.matrix.columns,
        ),
    )


__all__ = [
    "compute_nullspace",
    "compute_rank",
    "compute_rref",
]
