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
        rank=rank(matrix),
        prime=request.prime,
        entries=request.entries,
        columns=request.columns,
    )


def compute_rref(request: RrefRequest) -> RrefResult:
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
        pivot_columns=pivot_columns,
        computed_matrix=PrimeFieldMatrix(
            prime=request.prime, entries=rref_rows, columns=request.columns
        ),
    )


def compute_nullspace(request: NullspaceRequest) -> NullspaceResult:
    matrix = PrimeFieldMatrix(
        prime=request.prime,
        entries=request.entries,
        columns=request.columns,
    )
    ns = nullspace(matrix)
    # Full-rank source: a single zero row keeps field context and width
    # explicit for downstream consumers.
    basis_rows = tuple(ns) if ns else (tuple(0 for _ in range(request.columns)),)
    return NullspaceResult(
        prime=request.prime,
        entries=request.entries,
        columns=request.columns,
        nullspace_rows=ns,
        basis_matrix=PrimeFieldMatrix(
            prime=request.prime,
            entries=basis_rows,
            columns=request.columns,
        ),
    )


__all__ = [
    "compute_nullspace",
    "compute_rank",
    "compute_rref",
]
