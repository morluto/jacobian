"""Domain-owned prime-field matrix operations."""

from __future__ import annotations

from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix
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


def _verify_rank_result(result: PrimeFieldMatrixRankResult) -> bool:
    """Deliberately recheck one independently supplied rank claim."""

    source = PrimeFieldMatrixRequest.model_validate(result.source.model_dump())
    return result.prime == source.matrix.prime and result.rank == _rank(source.matrix)


def _verify_rref_result(result: PrimeFieldRrefResult) -> bool:
    """Deliberately recheck one independently supplied RREF claim."""

    source = PrimeFieldMatrixRequest.model_validate(result.source.model_dump())
    rows, pivots = _rref(source.matrix)
    return (
        result.prime == source.matrix.prime
        and result.rref_matrix.entries == tuple(rows)
        and result.rref_matrix.prime == source.matrix.prime
        and result.rref_matrix.columns == source.matrix.columns
        and result.pivot_columns == tuple(pivots)
        and result.rank == len(pivots)
    )


def _verify_nullspace_result(result: PrimeFieldNullspaceResult) -> bool:
    """Deliberately recheck one independently supplied nullspace claim."""

    source = PrimeFieldMatrixRequest.model_validate(result.source.model_dump())
    basis = _nullspace(source.matrix)
    return (
        result.prime == source.matrix.prime
        and result.nullspace_matrix.entries == tuple(basis)
        and result.nullspace_matrix.prime == source.matrix.prime
        and result.nullspace_matrix.columns == source.matrix.columns
        and result.nullity == len(basis)
    )


__all__ = [
    "compute_nullspace",
    "compute_rank",
    "compute_rref",
]
