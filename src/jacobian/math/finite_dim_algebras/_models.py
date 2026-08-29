"""Typed wire contracts for finite-dimensional algebra operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

# Worst-case structure tensors (field_order=251, every coefficient 250) must fit
# the 10 MiB CanonicalLimits request envelope before Pydantic validation.
# Dimension 137 encodes to about 9.84 MiB; 138 exceeds 10 MiB.
MAX_DIM = 137
MAX_STRUCTURE_CONSTANT_ENTRIES = MAX_DIM**3
MAX_COMMUTATOR_ENTRIES = MAX_STRUCTURE_CONSTANT_ENTRIES
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
        from sympy import isprime

        if not isprime(self.field_order):
            raise _validation_error(
                "field_order_not_prime", "field_order must be prime"
            )
        n = self.dimension
        if n**3 > MAX_STRUCTURE_CONSTANT_ENTRIES:
            raise _validation_error(
                "structure_constant_budget",
                "structure-constant tensor exceeds the materialization budget",
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
    center_basis: tuple[tuple[int, ...], ...] = Field(max_length=MAX_DIM)
    dimension: int = Field(ge=1, le=MAX_DIM)
    center_dimension: int = Field(ge=0, le=MAX_DIM)

    @model_validator(mode="after")
    def require_complete_basis_shape(self) -> Self:
        if len(self.center_basis) != self.center_dimension:
            raise _validation_error(
                "center_dimension_mismatch",
                "center_dimension must equal the number of basis vectors",
            )
        if any(len(vector) != self.dimension for vector in self.center_basis):
            raise _validation_error(
                "center_basis_shape",
                "every center basis vector must match the algebra dimension",
            )
        if self.center_dimension * self.dimension > MAX_CENTER_BASIS_ENTRIES:
            raise _validation_error(
                "center_basis_budget",
                "center basis exceeds the exact output budget",
            )
        return self
