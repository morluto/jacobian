"""Typed wire contracts for finite-dimensional algebra operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix

# Bound dense structure-tensor materialization before Pydantic validation.
# Dimension 137 retains 2,571,353 scalar cells; the next dimension crosses the
# owner-local retained-cell envelope.
MAX_REQUEST_ENCODING_DIM = 137

# Dense commutator matrix is n^2-by-n. Gaussian elimination / nullspace work is
# rows * columns * min(rows, columns) = n^4. This conservative ledger is the
# same 1024^3 cubic envelope already admitted for FLINT nmod_mat on dense
# prime-field matrices; the n^3 structure-constant budget does not imply it.
MAX_COMMUTATOR_ELIMINATION_WORK = 1_073_741_824


def commutator_elimination_work(dimension: int) -> int:
    """Return dense n^2-by-n commutator Gaussian-elimination work."""

    row_count = dimension * dimension
    column_count = dimension
    return row_count * column_count * min(row_count, column_count)


MAX_DIM = max(
    dimension
    for dimension in range(1, MAX_REQUEST_ENCODING_DIM + 1)
    if commutator_elimination_work(dimension) <= MAX_COMMUTATOR_ELIMINATION_WORK
)
MAX_STRUCTURE_CONSTANT_ENTRIES = MAX_DIM**3
MAX_CENTER_BASIS_ENTRIES = MAX_DIM * MAX_DIM


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by finite-dimensional algebras."""

    return PydanticCustomError(f"finite_dim_algebra.{reason}", message)


class StructureConstants(StrictModel):
    """Structure constants ``c[i][j][k]`` for a finite-dimensional algebra.

    The algebra has basis ``e_0, ..., e_{n-1}`` over the prime field ``F_q`` and

    ``e_i * e_j = sum_k c[i][j][k] e_k``

    where every ``c[i][j][k]`` is a canonical residue in ``{0, ..., q - 1}``.
    """

    dimension: int = Field(ge=1, le=MAX_DIM)
    field_order: int = Field(ge=2, le=251)
    multiplication: tuple[tuple[tuple[int, ...], ...], ...] = Field(
        min_length=1, max_length=MAX_DIM
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        n = self.dimension
        if n**3 > MAX_STRUCTURE_CONSTANT_ENTRIES:
            raise _validation_error(
                "structure_constant_budget",
                "structure-constant tensor exceeds the materialization budget",
            )
        if commutator_elimination_work(n) > MAX_COMMUTATOR_ELIMINATION_WORK:
            raise _validation_error(
                "commutator_elimination_work",
                "commutator nullspace exceeds the exact elimination-work budget",
            )
        if len(self.multiplication) != n:
            raise _validation_error(
                "multiplication_outer_dimension",
                "multiplication must have dimension rows",
            )
        for row in self.multiplication:
            if len(row) != n:
                raise _validation_error(
                    "multiplication_not_square",
                    "multiplication must be square in the first two indices",
                )
            for inner in row:
                if len(inner) != n:
                    raise _validation_error(
                        "multiplication_inner_dimension",
                        "multiplication must be a 3-index tensor c[i][j][k]",
                    )
                if any(not 0 <= v < self.field_order for v in inner):
                    raise _validation_error(
                        "multiplication_noncanonical_residue",
                        "entries must be canonical field residues",
                    )
        return self


# Requests


class CenterRequest(StrictModel):
    algebra: StructureConstants


# Results


class CenterResult(StrictModel):
    """A claimed center basis on the retained algebra's prime-field coordinates."""

    algebra: StructureConstants
    basis_matrix: PrimeFieldMatrix

    @property
    def dimension(self) -> int:
        return self.algebra.dimension

    @property
    def center_dimension(self) -> int:
        return len(self.basis_matrix.entries)

    @property
    def center_basis(self) -> tuple[tuple[int, ...], ...]:
        return self.basis_matrix.entries

    @model_validator(mode="after")
    def require_complete_basis_shape(self) -> Self:
        if (
            self.basis_matrix.prime != self.algebra.field_order
            or self.basis_matrix.columns != self.algebra.dimension
        ):
            raise _validation_error(
                "center_parent_axis",
                "center basis must retain the algebra field and coordinate axis",
            )
        if self.center_dimension > self.dimension:
            raise _validation_error(
                "center_basis_budget",
                "center basis cannot exceed the ambient dimension",
            )
        return self
