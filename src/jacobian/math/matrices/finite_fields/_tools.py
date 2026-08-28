"""Prime-field matrix operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.matrices.finite_fields import operations as native
from jacobian.math.matrices.finite_fields._models import (
    PrimeFieldMatrixRankResult,
    PrimeFieldMatrixRequest,
    PrimeFieldNullspaceResult,
    PrimeFieldRrefResult,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def compute_rank(request: PrimeFieldMatrixRequest) -> PrimeFieldMatrixRankResult:
    return PrimeFieldMatrixRankResult._from_kernel(
        request, rank=native.matrix_rank(request.matrix)
    )


def compute_rref(request: PrimeFieldMatrixRequest) -> PrimeFieldRrefResult:
    rref_rows, pivot_columns = native.matrix_rref(request.matrix)
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
    basis = native.matrix_nullspace(request.matrix)
    return PrimeFieldNullspaceResult._from_kernel(
        request,
        nullspace_matrix=PrimeFieldMatrix(
            prime=request.matrix.prime,
            entries=tuple(basis),
            columns=request.matrix.columns,
        ),
    )


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "prime_field.matrix.rank.compute",
        "Compute matrix rank over GF(p)",
        "Compute the exact rank of a bounded integer matrix over the "
        "prime field GF(p). The prime is supplied explicitly so that "
        "characteristic-dependent rank is always unambiguous.",
        PrimeFieldMatrixRequest,
        PrimeFieldMatrixRankResult,
        compute_rank,
        "linear-algebra",
        "finite-field",
        "exact",
        examples=(
            example(
                "rank_gf2",
                "Compute the rank of [[1,0,1],[0,1,1],[1,1,0]] over GF(2); "
                "the entries must be canonical residues in [0, prime).",
                {
                    "matrix": {
                        "prime": 2,
                        "entries": [[1, 0, 1], [0, 1, 1], [1, 1, 0]],
                        "columns": 3,
                    }
                },
            ),
        ),
    ),
    _op(
        "prime_field.matrix.rref.compute",
        "Compute RREF over GF(p)",
        "Compute the reduced row-echelon form of a bounded integer matrix "
        "over GF(p), with pivot columns and rank. The prime is supplied "
        "explicitly so that the field characteristic is always unambiguous.",
        PrimeFieldMatrixRequest,
        PrimeFieldRrefResult,
        compute_rref,
        "linear-algebra",
        "finite-field",
        "exact",
        examples=(
            example(
                "rref_gf2",
                "Compute the RREF of [[1,0,1],[0,1,1],[1,1,0]] over GF(2); "
                "the entries must be canonical residues in [0, prime).",
                {
                    "matrix": {
                        "prime": 2,
                        "entries": [[1, 0, 1], [0, 1, 1], [1, 1, 0]],
                        "columns": 3,
                    }
                },
            ),
        ),
    ),
    _op(
        "prime_field.matrix.nullspace.compute",
        "Compute nullspace basis over GF(p)",
        "Compute a deterministic basis for the right nullspace of a bounded "
        "integer matrix over GF(p). The prime is supplied explicitly so "
        "that the field characteristic is always unambiguous.",
        PrimeFieldMatrixRequest,
        PrimeFieldNullspaceResult,
        compute_nullspace,
        "linear-algebra",
        "finite-field",
        "exact",
        examples=(
            example(
                "nullspace_gf2",
                "Compute the nullspace of [[1,0,1],[0,1,1]] over GF(2); "
                "the entries must be canonical residues in [0, prime).",
                {
                    "matrix": {
                        "prime": 2,
                        "entries": [[1, 0, 1], [0, 1, 1]],
                        "columns": 3,
                    }
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
