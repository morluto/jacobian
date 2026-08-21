"""Domain-owned prime-field matrix operations."""

from __future__ import annotations

from jacobian.math.prime_field_matrix._models import (
    PrimeFieldMatrixRequest,
    PrimeFieldMatrixRankResult,
    PrimeFieldNullspaceResult,
    PrimeFieldRrefResult,
)
from jacobian.math.prime_field_linear_algebra import (
    PrimeFieldMatrix,
    nullspace as _nullspace,
    rank as _rank,
    rref as _rref,
)


def _to_kernel(request: PrimeFieldMatrixRequest) -> PrimeFieldMatrix:
    return PrimeFieldMatrix(
        prime=request.prime,
        entries=request.entries,
        columns=len(request.entries[0]),
    )


def compute_rank(request: PrimeFieldMatrixRequest) -> PrimeFieldMatrixRankResult:
    """Compute the rank of a matrix over GF(p)."""
    matrix = _to_kernel(request)
    return PrimeFieldMatrixRankResult(
        prime=request.prime,
        rows=len(request.entries),
        columns=len(request.entries[0]),
        rank=_rank(matrix),
    )


def compute_rref(request: PrimeFieldMatrixRequest) -> PrimeFieldRrefResult:
    """Compute the reduced row-echelon form of a matrix over GF(p)."""
    matrix = _to_kernel(request)
    rref_rows, pivot_columns = _rref(matrix)
    return PrimeFieldRrefResult(
        prime=request.prime,
        rows=len(request.entries),
        columns=len(request.entries[0]),
        rref=rref_rows,
        pivot_columns=pivot_columns,
        rank=len(pivot_columns),
    )


def compute_nullspace(request: PrimeFieldMatrixRequest) -> PrimeFieldNullspaceResult:
    """Compute a basis for the right nullspace of a matrix over GF(p)."""
    matrix = _to_kernel(request)
    basis = _nullspace(matrix)
    return PrimeFieldNullspaceResult(
        prime=request.prime,
        columns=len(request.entries[0]),
        nullspace=basis,
        nullity=len(basis),
    )


__all__ = [
    "compute_rank",
    "compute_rref",
    "compute_nullspace",
]
