"""Wire adapters for prime-field matrix operations."""

from jacobian.math.prime_field_linear_algebra import nullspace, rank, rref
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
    from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

    reduced_rows, pivot_columns = rref(request.matrix)
    reduced = PrimeFieldMatrix(
        prime=request.matrix.prime,
        entries=reduced_rows,
        columns=request.matrix.columns,
    )
    return RrefResult(
        matrix=request.matrix,
        rref=reduced,
        pivot_columns=pivot_columns,
    )


def compute_nullspace(request: NullspaceRequest) -> NullspaceResult:
    return NullspaceResult(
        matrix=request.matrix,
        nullspace_rows=nullspace(request.matrix),
    )


__all__ = [
    "compute_nullspace",
    "compute_rank",
    "compute_rref",
]
