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


def _matrix(request: RankRequest) -> PrimeFieldMatrix:
    return PrimeFieldMatrix(
        prime=request.prime,
        entries=request.entries,
        columns=request.columns,
    )


def compute_rank(request: RankRequest) -> RankResult:
    matrix = _matrix(request)
    return RankResult(
        entries=request.entries,
        columns=request.columns,
        rank=rank(matrix),
        prime=request.prime,
    )


def compute_rref(request: RrefRequest) -> RrefResult:
    matrix = PrimeFieldMatrix(
        prime=request.prime,
        entries=request.entries,
        columns=request.columns,
    )
    reduced_rows, pivot_columns = rref(matrix)
    return RrefResult(
        prime=request.prime,
        entries=request.entries,
        columns=request.columns,
        reduced_matrix=PrimeFieldMatrix(
            prime=request.prime,
            entries=reduced_rows,
            columns=request.columns,
        ),
        pivot_columns=pivot_columns,
    )


def compute_nullspace(request: NullspaceRequest) -> NullspaceResult:
    matrix = PrimeFieldMatrix(
        prime=request.prime,
        entries=request.entries,
        columns=request.columns,
    )
    ns = nullspace(matrix)
    nullspace_matrix = PrimeFieldMatrix(
        prime=matrix.prime,
        entries=tuple(tuple(row) for row in ns),
        columns=matrix.columns,
    )
    return NullspaceResult(
        prime=request.prime,
        entries=request.entries,
        columns=request.columns,
        nullspace_matrix=nullspace_matrix,
    )


__all__ = [
    "compute_nullspace",
    "compute_rank",
    "compute_rref",
]
