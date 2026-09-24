"""Prime-field matrix operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.matrices.finite_fields import operations as native
from jacobian.math.matrices.finite_fields._models import (
    PrimeFieldMatrixRankResult,
    PrimeFieldMatrixRequest,
    PrimeFieldNullspaceResult,
    PrimeFieldRrefResult,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix
from jacobian.math.matrices.finite_fields.quotient_spaces import (
    PrimeFieldQuotientRequest,
    PrimeFieldQuotientSpace,
    PrimeFieldQuotientVector,
    PrimeFieldVectorProjectionRequest,
)


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


def compute_quotient_space(
    request: PrimeFieldQuotientRequest,
) -> PrimeFieldQuotientSpace:
    return native.quotient_space(request)


def project_quotient_vector(
    request: PrimeFieldVectorProjectionRequest,
) -> PrimeFieldQuotientVector:
    return native.project_quotient_vector(request)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="prime_field.matrix.rank.compute",
        title="Compute matrix rank over GF(p)",
        description="Compute the exact rank of a bounded integer matrix over the "
        "prime field GF(p). The prime is supplied explicitly so that "
        "characteristic-dependent rank is always unambiguous.",
        request_type=PrimeFieldMatrixRequest,
        result_type=PrimeFieldMatrixRankResult,
        run=compute_rank,
        tags=("linear-algebra", "finite-field", "exact"),
        examples=(
            OperationExample(
                name="rank_gf2",
                description="Compute the rank of [[1,0,1],[0,1,1],[1,1,0]] over GF(2); "
                "the entries must be canonical residues in [0, prime).",
                input={
                    "matrix": {
                        "prime": 2,
                        "entries": [[1, 0, 1], [0, 1, 1], [1, 1, 0]],
                        "columns": 3,
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="prime_field.matrix.rref.compute",
        title="Compute RREF over GF(p)",
        description="Compute the reduced row-echelon form of a bounded integer matrix "
        "over GF(p), with pivot columns and rank. The prime is supplied "
        "explicitly so that the field characteristic is always unambiguous.",
        request_type=PrimeFieldMatrixRequest,
        result_type=PrimeFieldRrefResult,
        run=compute_rref,
        discovery_terms=(
            "solve linear system over GF(2)",
            "solve Ax=b over GF(p)",
            "prime field linear equations",
            "augmented matrix consistency",
            "particular solution nullspace basis",
        ),
        tags=("linear-algebra", "finite-field", "exact"),
        examples=(
            OperationExample(
                name="rref_gf2",
                description="Compute the RREF of [[1,0,1],[0,1,1],[1,1,0]] over GF(2); "
                "the entries must be canonical residues in [0, prime).",
                input={
                    "matrix": {
                        "prime": 2,
                        "entries": [[1, 0, 1], [0, 1, 1], [1, 1, 0]],
                        "columns": 3,
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="prime_field.matrix.nullspace.compute",
        title="Compute nullspace basis over GF(p)",
        description="Compute a deterministic basis for the right nullspace of a bounded "
        "integer matrix over GF(p). The prime is supplied explicitly so "
        "that the field characteristic is always unambiguous.",
        request_type=PrimeFieldMatrixRequest,
        result_type=PrimeFieldNullspaceResult,
        run=compute_nullspace,
        discovery_terms=(
            "GF(2) linear system nullspace basis",
            "GF(p) left nullspace inconsistency certificate",
            "prime field homogeneous system",
        ),
        tags=("linear-algebra", "finite-field", "exact"),
        examples=(
            OperationExample(
                name="nullspace_gf2",
                description="Compute the nullspace of [[1,0,1],[0,1,1]] over GF(2); "
                "the entries must be canonical residues in [0, prime).",
                input={
                    "matrix": {
                        "prime": 2,
                        "entries": [[1, 0, 1], [0, 1, 1]],
                        "columns": 3,
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="prime_field.vector_space.quotient.compute",
        title="Construct a prime-field quotient space",
        description=(
            "Construct GF(p)^n/W from bounded subspace generators. Return a "
            "source-bound quotient basis and the exact projection matrix from "
            "ambient coordinates to quotient coordinates."
        ),
        request_type=PrimeFieldQuotientRequest,
        result_type=PrimeFieldQuotientSpace,
        run=compute_quotient_space,
        tags=("linear-algebra", "finite-field", "quotient", "exact"),
        examples=(
            OperationExample(
                name="coordinate_quotient_gf3",
                description=(
                    "Form GF(3)^2 / span((1,1)); the returned map sends (x,y) "
                    "to x-y in the basis represented by (1,0)."
                ),
                input={
                    "subspace": {
                        "prime": 3,
                        "ambient_dimension": 2,
                        "generators": [[1, 1]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="prime_field.vector_space.quotient.project",
        title="Project a vector to a prime-field quotient",
        description=(
            "Apply a source-bound GF(p) quotient projection to an ambient vector "
            "and return quotient coordinates with the same quotient parent."
        ),
        request_type=PrimeFieldVectorProjectionRequest,
        result_type=PrimeFieldQuotientVector,
        run=project_quotient_vector,
        tags=("linear-algebra", "finite-field", "quotient", "exact"),
        examples=(
            OperationExample(
                name="project_gf3_vector",
                description="Project the ambient vector (2,0) using the quotient above.",
                input={
                    "quotient": {
                        "source": {
                            "prime": 3,
                            "ambient_dimension": 2,
                            "generators": [[1, 1]],
                        },
                        "quotient_basis": [[1, 0]],
                        "projection": {
                            "prime": 3,
                            "entries": [[1, 2]],
                            "columns": 2,
                        },
                    },
                    "vector": [2, 0],
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
