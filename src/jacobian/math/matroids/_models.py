"""Typed wire contracts for linear matroid operations over a finite field."""

from __future__ import annotations

from typing import Any, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

MAX_GROUND_SIZE = 32
"""Schema-visible cap on the ground-set cardinality (matrix columns)."""

MAX_PRIME = 2_147_483_647
"""Explicit conservative bound on the field prime before primality testing."""


class LinearMatroid(StrictModel):
    """A linear matroid over GF(p) represented by a canonical matrix.

    The ground set ``{0, ..., columns - 1}`` indexes the columns of the
    domain-owned ``PrimeFieldMatrix``; rank and closure derive from the
    column span. The empty matroid is admitted: with zero columns the row
    axis stays declared, the rank is exactly zero, and every closure is
    empty. The characteristic is bounded before construction so no accepted
    value performs unbounded primality work.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A linear matroid over GF(p) as the canonical "
                "`PrimeFieldMatrix`: the ground set indexes matrix columns, "
                "entries are canonical residues in [0, prime), and up to 32 "
                "columns. The empty matroid (zero columns) is admitted."
            )
        }
    )

    matrix: PrimeFieldMatrix

    @model_validator(mode="before")
    @classmethod
    def require_bounded_declared_prime(cls, data: Any) -> Any:
        if isinstance(data, dict):
            raw = data.get("matrix")
            prime = (
                raw.get("prime")
                if isinstance(raw, dict)
                else getattr(raw, "prime", None)
            )
            if isinstance(prime, int) and not 2 <= prime <= MAX_PRIME:
                raise ValueError(
                    f"field prime must lie in [2, {MAX_PRIME}] so validation "
                    "work stays bounded"
                )
        return data

    @property
    def ground_size(self) -> int:
        return self.matrix.columns


class MatroidClosureRequest(StrictModel):
    """Compute the closure of a subset in a linear matroid."""

    matroid: LinearMatroid
    subset: tuple[StrictInt, ...] = Field(default=(), max_length=MAX_GROUND_SIZE)

    @model_validator(mode="after")
    def require_valid_subset(self) -> Self:
        for idx in self.subset:
            if not (0 <= idx < self.matroid.ground_size):
                raise ValueError("subset indices must be in 0..n-1")
        if len(set(self.subset)) != len(self.subset):
            raise ValueError("subset indices must be distinct")
        return self


class MatroidClosureResult(MatroidClosureRequest):
    """The closure (flat) of a subset in a linear matroid."""

    closure: tuple[StrictInt, ...] = Field(default=(), max_length=MAX_GROUND_SIZE)
    rank: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_closure(self) -> Self:
        from jacobian.math.matroids._operations import (
            _closure_invariant,
        )

        expected_closure, subset_rank = _closure_invariant(
            self.matroid, list(self.subset)
        )
        if tuple(self.closure) != tuple(expected_closure):
            raise ValueError("closure must be the exact flat of the subset")
        if self.rank != subset_rank:
            raise ValueError("rank must be the rank of the requested subset")
        return self


__all__ = [
    "LinearMatroid",
    "MatroidClosureRequest",
    "MatroidClosureResult",
]
