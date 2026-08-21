"""Typed wire contracts for symmetric function operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

_MAX_PARTITION_SIZE = 100
_MAX_PARTITION_PARTS = 50


class IntegerPartition(StrictModel):
    """A partition of a positive integer as a weakly decreasing tuple."""

    parts: tuple[int, ...] = Field(min_length=0, max_length=_MAX_PARTITION_PARTS)

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
    conjugate: tuple[int, ...]


class SchurExpansionRequest(StrictModel):
    partition: IntegerPartition
    variables: tuple[str, ...] = Field(min_length=1, max_length=20)
    point: tuple[int, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def require_matching_dimensions(self) -> Self:
        if len(self.variables) != len(self.point):
            raise ValueError("variables and point must have the same length")
        return self


class SchurExpansionResult(StrictModel):
    value: int


__all__ = [
    "IntegerPartition",
    "PartitionConjugateResult",
    "PartitionRequest",
    "SchurExpansionRequest",
    "SchurExpansionResult",
]
