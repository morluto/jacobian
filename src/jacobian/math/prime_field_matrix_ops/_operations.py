"""Wire adapters for prime-field matrix operations."""

from jacobian.math.prime_field_linear_algebra import (
    PrimeFieldMatrix,
    nullspace,
    rank,
    rref,
)
from jacobian.math.prime_field_matrix_ops._models import (
    NullspaceRequest,
    NullspaceResult,
    RankRequest,
    RankResult,
    RrefRequest,
    RrefResult,
)


def compute_rank(request: RankRequest) -> RankResult:
    return RankResult(matrix=request.matrix, rank=rank(request.matrix))


def compute_rref(request: RrefRequest) -> RrefResult:
    rref_rows, pivot_columns = rref(request.matrix)
    rref_matrix = PrimeFieldMatrix(
        prime=request.matrix.prime,
        entries=rref_rows,
        columns=request.matrix.columns,
    )
    return RrefResult(
        matrix=request.matrix,
        rref_matrix=rref_matrix,
        pivot_columns=pivot_columns,
    )


def compute_nullspace(request: NullspaceRequest) -> NullspaceResult:
    rows = nullspace(request.matrix)
    nullspace_matrix = PrimeFieldMatrix(
        prime=request.matrix.prime,
        entries=tuple(tuple(row) for row in rows),
        columns=request.matrix.columns,
    )
    return NullspaceResult(
        matrix=request.matrix,
        nullspace_matrix=nullspace_matrix,
    )


__all__ = [
    "compute_nullspace",
    "compute_rank",
    "compute_rref",
]
