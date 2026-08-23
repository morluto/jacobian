"""Typed wire contracts for symmetric function operations."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel

_MAX_PARTITION_SIZE = 100
_MAX_PARTITION_PARTS = 50
_MAX_POINT_COORDINATE_DIGITS = 6
_MAX_POINT_COORDINATE_ABS = 10**_MAX_POINT_COORDINATE_DIGITS - 1
_MAX_SCHUR_RESULT_DIGITS = 4000

PointCoordinate = Annotated[
    int,
    Field(
        ge=-_MAX_POINT_COORDINATE_ABS,
        le=_MAX_POINT_COORDINATE_ABS,
        description=(
            "Canonical integer with at most "
            f"{_MAX_POINT_COORDINATE_DIGITS} decimal digits."
        ),
    ),
]
"""One bounded evaluation coordinate: ``abs(value) <= 10**6 - 1``."""


class IntegerPartition(StrictModel):
    """A partition of a positive integer as a weakly decreasing tuple.

    Parts must be positive and weakly decreasing, there are at most 50
    parts, and the total size (sum of the parts) is capped at 100.
    """

    parts: tuple[int, ...] = Field(
        min_length=0,
        max_length=_MAX_PARTITION_PARTS,
        description=(
            f"Positive weakly-decreasing parts with a total size (sum) of at "
            f"most {_MAX_PARTITION_SIZE}; at most {_MAX_PARTITION_PARTS} parts."
        ),
    )

    @model_validator(mode="after")
    def require_valid_partition(self) -> Self:
        if not self.parts:
            return self
        if any(p <= 0 for p in self.parts):
            raise ValueError("partition parts must be positive")
        if any(self.parts[i] < self.parts[i + 1] for i in range(len(self.parts) - 1)):
            raise ValueError("partition parts must be weakly decreasing")
        if sum(self.parts) > _MAX_PARTITION_SIZE:
            raise ValueError("partition size exceeds the supported bound")
        return self


class PartitionRequest(StrictModel):
    partition: IntegerPartition


class PartitionConjugateResult(StrictModel):
    conjugate: IntegerPartition


class SchurExpansionRequest(StrictModel):
    """Evaluate one Schur function at a bounded integer point.

    Preconditions published through this schema: ``variables`` and ``point``
    must have equal lengths in ``[1, 20]``, variable names must be distinct,
    each coordinate satisfies ``abs(coordinate) <= 999999``, and the partition
    total size is capped at 100.
    """

    partition: IntegerPartition
    variables: tuple[str, ...] = Field(
        min_length=1,
        max_length=20,
        description=(
            "Distinct variable names; the length must equal the length of "
            "point (between 1 and 20)."
        ),
        json_schema_extra={"uniqueItems": True},
    )
    point: tuple[PointCoordinate, ...] = Field(
        min_length=1,
        max_length=20,
        description=(
            "Integer evaluation coordinates, one per variable, each with at "
            f"most {_MAX_POINT_COORDINATE_DIGITS} decimal digits; the length "
            "must equal the length of variables."
        ),
    )

    @model_validator(mode="after")
    def require_matching_dimensions(self) -> Self:
        if len(self.variables) != len(self.point):
            raise ValueError("variables and point must have the same length")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("variables must be distinct (duplicate axis)")
        return self


class SchurExpansionResult(StrictModel):
    value: CanonicalInteger

    @model_validator(mode="after")
    def require_bounded_value(self) -> Self:
        if len(self.value.lstrip("-")) > _MAX_SCHUR_RESULT_DIGITS:
            raise ValueError("Schur value exceeds the output digit bound")
        return self


__all__ = [
    "IntegerPartition",
    "PartitionConjugateResult",
    "PartitionRequest",
    "SchurExpansionRequest",
    "SchurExpansionResult",
]
