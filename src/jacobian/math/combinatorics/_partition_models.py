"""Typed contracts and bounds for bounded integer-partition enumeration."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.combinatorics._models import _combinatorics_validation_error

MAX_PARTITION_N = 30
MAX_ENUMERATED_PARTITIONS = 10_000


class IntegerPartitionEnumerationRequest(StrictModel):
    """Enumerate every partition of n containing at most max_parts summands."""

    n: StrictInt = Field(ge=0, le=MAX_PARTITION_N)
    max_parts: StrictInt = Field(ge=1, le=MAX_PARTITION_N)


class IntegerPartitionEnumerationResult(StrictModel):
    """Complete canonical partition enumeration for one bounded request."""

    n: StrictInt = Field(ge=0, le=MAX_PARTITION_N)
    max_parts: StrictInt = Field(ge=1, le=MAX_PARTITION_N)
    partitions: tuple[tuple[StrictInt, ...], ...] = Field(
        max_length=MAX_ENUMERATED_PARTITIONS
    )

    @model_validator(mode="after")
    def require_canonical_complete_items(self) -> Self:
        previous: tuple[int, ...] | None = None
        for partition in self.partitions:
            if len(partition) > self.max_parts:
                raise _combinatorics_validation_error("partition exceeds max_parts")
            if any(part <= 0 for part in partition):
                raise _combinatorics_validation_error(
                    "partition parts must be positive"
                )
            if tuple(sorted(partition, reverse=True)) != partition:
                raise _combinatorics_validation_error(
                    "partition parts must be nonincreasing"
                )
            if sum(partition) != self.n:
                raise _combinatorics_validation_error("partition parts must sum to n")
            if previous is not None and previous <= partition:
                raise _combinatorics_validation_error(
                    "partitions must be unique in descending lexicographic order"
                )
            previous = tuple(partition)
        if self.n == 0 and self.partitions != ((),):
            raise _combinatorics_validation_error(
                "zero has exactly one empty partition"
            )
        return self
