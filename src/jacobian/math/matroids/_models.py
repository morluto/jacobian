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
        # Running a before validator moves field validation into Python
        # mode, where decoded JSON arrays no longer coerce to the declared
        # tuple shapes; normalize entry rows to tuples on a copied path so
        # JSON invocation keeps working while stored values stay canonical.
        if isinstance(data, dict):
            raw = data.get("matrix")
            if isinstance(raw, dict):
                prime = raw.get("prime")
                entries = raw.get("entries")
                if isinstance(entries, list):
                    matrix = dict(raw)
                    matrix["entries"] = tuple(
                        tuple(row) if isinstance(row, list) else row for row in entries
                    )
                    data = {**data, "matrix": matrix}
            else:
                prime = getattr(raw, "prime", None)
            if isinstance(prime, int) and not 2 <= prime <= MAX_PRIME:
                raise ValueError(
                    f"field prime must lie in [2, {MAX_PRIME}] so validation "
                    "work stays bounded"
                )
        return data

    @model_validator(mode="after")
    def require_bounded_ground_set(self) -> Self:
        if self.matrix.columns > MAX_GROUND_SIZE:
            raise ValueError(
                f"ground set must hold at most {MAX_GROUND_SIZE} elements, "
                f"got {self.matrix.columns}"
            )
        return self

    @property
    def ground_size(self) -> int:
        return self.matrix.columns


def validate_subset_indices(matroid: LinearMatroid, subset: Any) -> None:
    """Shared closure-subset admission: in-range and distinct indices.

    Used by both the wire request model and the native entry point so a
    direct kernel call can never admit indices the wire path rejects
    (negative indexing must not select columns).
    """
    for idx in subset:
        if not (0 <= idx < matroid.ground_size):
            raise ValueError("subset indices must be in 0..n-1")
    if len(set(subset)) != len(subset):
        raise ValueError("subset indices must be distinct")


class MatroidClosureRequest(StrictModel):
    """Compute the closure of a subset in a linear matroid."""

    matroid: LinearMatroid
    subset: tuple[StrictInt, ...] = Field(default=(), max_length=MAX_GROUND_SIZE)

    @model_validator(mode="after")
    def require_valid_subset(self) -> Self:
        validate_subset_indices(self.matroid, self.subset)
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
    "validate_subset_indices",
]
