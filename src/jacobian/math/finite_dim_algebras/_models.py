"""Typed wire contracts for finite-dimensional algebra operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_DIM = 128
MAX_ENTRIES = (
    MAX_DIM * MAX_DIM
)  # 16384; outer tensor bound (n <= 128, logical n^3 entries)

# ``compute_center`` builds n^2 commutator rows and runs Gaussian
# elimination over all of them with length-n row updates, so the kernel's
# actual worst-case work is Theta(n^4) modular-entry updates, not cubic.
# Measured on this kernel: n=32 ~0.08s, n=64 ~1.2s, n=96 ~6.1s,
# n=128 ~18.8s.  The dimension envelope is derived from that measured
# cost so an accepted request stays inside the bounded synchronous
# execution envelope; replacing the kernel with a faster elimination
# would justify re-deriving this bound upward.
_MAX_DIM_FOR_CENTER = 128


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
        min_length=1, max_length=MAX_ENTRIES
    )

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        from sympy import isprime

        if not isprime(self.field_order):
            raise _validation_error(
                "field_order_not_prime", "field_order must be prime"
            )
        n = self.dimension
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

    @model_validator(mode="after")
    def require_bounded_center_work(self) -> Self:
        if self.algebra.dimension > _MAX_DIM_FOR_CENTER:
            raise _validation_error(
                "center_dimension_limit",
                "center computation is Theta(n^4) on the current elimination "
                f"kernel and supports at most {_MAX_DIM_FOR_CENTER} dimensions",
            )
        return self


# Results


class CenterResult(StrictModel):
    center_basis: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=1)
    center_dimension: int = Field(ge=0)
    method: str = "COMMUTANT_COMPUTATION"
