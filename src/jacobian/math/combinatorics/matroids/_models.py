"""Typed wire contracts for linear matroid operations over a finite field."""

from __future__ import annotations

from typing import Any, Self

from pydantic import ConfigDict, Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix

MAX_GROUND_SIZE = 256
"""Schema-visible cap on the ground-set cardinality (matrix columns)."""

MAX_REPRESENTATION_ROWS = 256
"""Preserved row envelope for matroid representation and witness work."""

MAX_PRIME = 2_147_483_647
"""Explicit conservative bound on the field prime before primality testing."""


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by matroid contracts."""

    return PydanticCustomError(f"matroid.{reason}", message)


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
                "entries are canonical residues in [0, prime), and up to 256 "
                "columns. The empty matroid (zero columns) is admitted."
            )
        }
    )

    matrix: PrimeFieldMatrix

    @model_validator(mode="before")
    @classmethod
    def require_bounded_declared_prime(cls, data: Any) -> Any:
        data = canonicalize_json_containers(data)
        # Shared strict-JSON container canonicalization above keeps the
        # nested matrix entry rows in their declared tuple shape.
        if isinstance(data, dict):
            raw = data.get("matrix")
            if isinstance(raw, dict):
                prime = raw.get("prime")
            else:
                prime = getattr(raw, "prime", None)
            if isinstance(prime, int) and not 2 <= prime <= MAX_PRIME:
                raise _validation_error(
                    "field_prime.bound",
                    f"field prime must lie in [2, {MAX_PRIME}] so validation "
                    "work stays bounded",
                )
        return data

    @model_validator(mode="after")
    def require_bounded_ground_set(self) -> Self:
        if len(self.matrix.entries) > MAX_REPRESENTATION_ROWS:
            raise _validation_error(
                "representation_rows.bound",
                "matroid representation must have at most "
                f"{MAX_REPRESENTATION_ROWS} rows",
            )
        if self.matrix.columns > MAX_GROUND_SIZE:
            raise _validation_error(
                "ground_set.bound",
                f"ground set must hold at most {MAX_GROUND_SIZE} elements, "
                f"got {self.matrix.columns}",
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

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute the closure of a distinct subset of the ground set of "
                "a bounded linear matroid. Ground-set elements are the columns "
                "of `matroid.matrix`, indexed from 0 through "
                "`matroid.matrix.columns - 1`."
            )
        }
    )

    matroid: LinearMatroid
    subset: tuple[StrictInt, ...] = Field(
        default=(),
        max_length=MAX_GROUND_SIZE,
        description=(
            "Distinct ground-set indices. Every index must lie in "
            "0..matroid.matrix.columns-1; at most "
            f"{MAX_GROUND_SIZE} indices are admitted."
        ),
        json_schema_extra={"uniqueItems": True},
    )


class MatroidClosureResult(MatroidClosureRequest):
    """The claimed closure (flat) of a subset in a linear matroid.

    Deserialization establishes only the retained request and bounded canonical
    result shape. Kernel output uses ``_from_kernel`` after its trusted bounded
    computation.
    """

    closure: tuple[StrictInt, ...] = Field(
        default=(),
        max_length=MAX_GROUND_SIZE,
        description=(
            "The complete flat spanned by `subset`, as distinct ground-set "
            "indices in increasing order."
        ),
    )
    rank: StrictInt = Field(
        ge=0,
        description="The exact rank of `subset` in the declared linear matroid.",
    )

    @model_validator(mode="after")
    def require_bounded_canonical_claim(self) -> Self:
        if self.closure != tuple(sorted(set(self.closure))):
            raise _validation_error(
                "closure.canonical",
                "closure indices must be distinct and in increasing order",
            )
        if any(not 0 <= index < self.matroid.ground_size for index in self.closure):
            raise _validation_error(
                "closure.indices",
                "closure indices must be in 0..matroid.matrix.columns-1",
            )
        if self.rank > len(self.subset):
            raise _validation_error(
                "rank.bound",
                "rank cannot exceed the number of selected ground-set elements",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        matroid: LinearMatroid,
        subset: tuple[int, ...],
        closure: tuple[int, ...],
        rank: int,
    ) -> Self:
        """Construct trusted output of the owner-local closure kernel."""

        return cls.model_construct(
            matroid=matroid,
            subset=subset,
            closure=closure,
            rank=rank,
        )


__all__ = [
    "LinearMatroid",
    "MatroidClosureRequest",
    "MatroidClosureResult",
    "validate_subset_indices",
]
