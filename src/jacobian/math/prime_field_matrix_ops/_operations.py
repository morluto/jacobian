"""Wire adapters for prime-field matrix operations."""

from __future__ import annotations

from jacobian.math.prime_field_linear_algebra import (
    PrimeFieldMatrix,
    nullspace,
    rank,
    rref,
)
from jacobian.math.prime_field_matrix_ops._models import (
    NullspaceRequest,
    NullspaceResult,
    PrimeFieldMatrixValue,
    RankRequest,
    RankResult,
    RrefRequest,
    RrefResult,
)


def _matrix(request: RankRequest | PrimeFieldMatrixValue) -> PrimeFieldMatrix:
    return PrimeFieldMatrix(
        prime=request.prime,
        entries=request.entries,
        columns=request.columns,
    )


def compute_rank(request: RankRequest | PrimeFieldMatrixValue) -> RankResult:
    # Accept a reusable PrimeFieldMatrixValue without reconstruction.
    if isinstance(request, PrimeFieldMatrixValue):
        request = RankRequest(
            prime=request.prime, entries=request.entries, columns=request.columns
        )
    matrix = _matrix(request)
    return RankResult(
        entries=request.entries,
        columns=request.columns,
        rank=rank(matrix),
        prime=request.prime,
    )


def compute_rref(request: RrefRequest | PrimeFieldMatrixValue) -> RrefResult:
    if isinstance(request, PrimeFieldMatrixValue):
        request = RrefRequest(
            prime=request.prime, entries=request.entries, columns=request.columns
        )
    matrix = PrimeFieldMatrix(
        prime=request.prime,
        entries=request.entries,
        columns=request.columns,
    )
    rref_rows, pivot_columns = rref(matrix)
    return RrefResult(
        prime=request.prime,
        entries=request.entries,
        columns=request.columns,
        rref_rows=rref_rows,
        rref=PrimeFieldMatrixValue(
            prime=request.prime, entries=rref_rows, columns=request.columns
        ),
        pivot_columns=pivot_columns,
    )


def compute_nullspace(
    request: NullspaceRequest | PrimeFieldMatrixValue,
) -> NullspaceResult:
    if isinstance(request, PrimeFieldMatrixValue):
        request = NullspaceRequest(
            prime=request.prime, entries=request.entries, columns=request.columns
        )
    matrix = PrimeFieldMatrix(
        prime=request.prime,
        entries=request.entries,
        columns=request.columns,
    )
    ns = nullspace(matrix)
    return NullspaceResult(
        prime=request.prime,
        entries=request.entries,
        columns=request.columns,
        nullspace_rows=ns,
        nullspace=PrimeFieldMatrixValue(
            prime=request.prime, entries=ns, columns=request.columns
        ),
    )


__all__ = [
    "compute_nullspace",
    "compute_rank",
    "compute_rref",
]
