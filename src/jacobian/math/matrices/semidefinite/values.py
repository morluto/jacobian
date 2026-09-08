"""Rational affine sections of a positive-semidefinite cone."""

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.matrices.values import RationalMatrix

MAX_SEMIDEFINITE_CELLS = 131_072


class RationalSemidefiniteSystem(StrictModel):
    """The equalities tr(A_i X)=b_i with X PSD in the ordered matrix axes.

    ``matrices`` and ``rhs`` must have the same length. Every matrix must be
    symmetric and exactly ``order`` by ``order``.
    """

    order: int = Field(
        ge=0,
        le=8192,
        description="Common axis length; every constraint matrix is order-by-order.",
    )
    matrices: tuple[RationalMatrix, ...] = Field(
        max_length=8192,
        description=(
            "Symmetric order-by-order QQ constraint matrices, one per equality "
            "and in the same order as rhs."
        ),
    )
    rhs: tuple[CanonicalRational, ...] = Field(
        max_length=8192,
        description="Right-hand sides; length must match matrices.",
    )

    @model_validator(mode="after")
    def require_symmetric_equalities(self) -> Self:
        if len(self.matrices) != len(self.rhs):
            raise ValueError("one right-hand side is required per equality")
        for matrix in self.matrices:
            if matrix.row_count != self.order or matrix.column_count != self.order:
                raise ValueError("constraint matrices must have the declared order")
            if any(
                matrix.entries[i][j] != matrix.entries[j][i]
                for i in range(self.order)
                for j in range(i)
            ):
                raise ValueError("constraint matrices must be symmetric")
        return self


class SemidefiniteFaceReduction(StrictModel):
    """Equivalent PSD equalities under X = embedding Y embedding^T.

    Parsing checks dimensions only. A consumer relying on an authored exposing
    relation must recognize it by calling reduce_exposed_face on source and
    multipliers; structural parsing does not establish feasible-set equivalence.
    """

    source: RationalSemidefiniteSystem
    multipliers: tuple[CanonicalRational, ...] = Field(max_length=8192)
    exposing_matrix: RationalMatrix
    embedding: RationalMatrix
    reduced: RationalSemidefiniteSystem

    @model_validator(mode="after")
    def require_shapes(self) -> Self:
        n = self.source.order
        if len(self.multipliers) != len(self.source.matrices):
            raise ValueError("multipliers must index source equalities")
        if (self.exposing_matrix.row_count, self.exposing_matrix.column_count) != (
            n,
            n,
        ):
            raise ValueError("exposing matrix must use the source axes")
        if (self.embedding.row_count, self.embedding.column_count) != (
            n,
            self.reduced.order,
        ):
            raise ValueError("embedding must map reduced axes into source axes")
        if self.reduced.order >= n:
            raise ValueError("a proper exposed face has strictly smaller order")
        if self.reduced.rhs != self.source.rhs:
            raise ValueError("face reduction preserves every right-hand side")
        return self
