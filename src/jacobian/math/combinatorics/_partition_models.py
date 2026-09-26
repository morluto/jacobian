"""Typed contracts and bounds for bounded integer-partition enumeration."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, conint, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics._models import _combinatorics_validation_error
from jacobian.math.combinatorics.symmetric_functions.values import (
    MAX_PARTITION_SIZE,
    IntegerPartition,
)

MAX_PARTITION_N = 30
MAX_ENUMERATED_PARTITIONS = 10_000
MAX_PARTITION_ITEM = 2**53 - 1
PartitionItem = conint(strict=True, ge=-MAX_PARTITION_ITEM, le=MAX_PARTITION_ITEM)


class PartitionCheckRequest(StrictModel):
    """Classify one bounded raw sequence as a partition or an obstruction."""

    parts: tuple[PartitionItem, ...] = Field(
        min_length=0,
        max_length=MAX_PARTITION_SIZE,
        description=(
            "A bounded sequence of exact JSON-safe integers. The size bound "
            f"applies when no partition obstruction is present ({MAX_PARTITION_SIZE})."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def preflight_raw_parts(cls, value: object) -> object:
        """Bound the caller sequence before Pydantic builds its tuple value."""

        if not isinstance(value, dict) or "parts" not in value:
            return value
        parts = value["parts"]
        if type(parts) not in (list, tuple):
            return value
        if len(parts) > MAX_PARTITION_SIZE:
            raise PydanticCustomError(
                "combinatorics.partition_candidate_length",
                "candidate has more parts than the supported bound",
            )
        if all(type(part) is int for part in parts):
            if any(abs(part) > MAX_PARTITION_ITEM for part in parts):
                raise PydanticCustomError(
                    "combinatorics.partition_candidate_integer",
                    "candidate part exceeds the exact JSON integer bound",
                )
        if type(parts) is list:
            admitted = dict(value)
            admitted["parts"] = tuple(parts)
            return admitted
        return value


class PartitionFound(StrictModel):
    """A canonical partition with its requested derived Ferrers data."""

    kind: Literal["PARTITION"] = "PARTITION"
    partition: IntegerPartition
    size: StrictInt = Field(ge=0, le=MAX_PARTITION_SIZE)
    length: StrictInt = Field(ge=0, le=MAX_PARTITION_SIZE)
    conjugate: IntegerPartition
    cells: tuple[tuple[StrictInt, StrictInt], ...] = Field(
        max_length=MAX_PARTITION_SIZE,
        description="One-based Ferrers cells in row-major order.",
    )

    @classmethod
    def _from_checked(
        cls,
        *,
        partition: IntegerPartition,
        conjugate: IntegerPartition,
        cells: tuple[tuple[int, int], ...],
    ) -> PartitionFound:
        """Build the result from derived data computed by the owning operation."""
        return cls.model_construct(
            partition=partition,
            size=sum(partition.parts),
            length=len(partition.parts),
            conjugate=conjugate,
            cells=cells,
        )

class NonpositivePartObstruction(StrictModel):
    """First candidate position whose part is nonpositive."""

    kind: Literal["NONPOSITIVE_PART"] = "NONPOSITIVE_PART"
    index: StrictInt = Field(ge=0, le=MAX_PARTITION_SIZE - 1)
    value: StrictInt = Field(le=0)


class IncreasingPartsObstruction(StrictModel):
    """First adjacent increase, indexed at its right-hand part."""

    kind: Literal["INCREASING_ADJACENT_PARTS"] = "INCREASING_ADJACENT_PARTS"
    index: StrictInt = Field(ge=1, le=MAX_PARTITION_SIZE - 1)
    previous_value: StrictInt = Field(ge=1)
    value: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def require_increase(self) -> Self:
        if self.previous_value >= self.value:
            raise _combinatorics_validation_error(
                "partition increase obstruction must be strict"
            )
        return self


PartitionObstruction = Annotated[
    NonpositivePartObstruction | IncreasingPartsObstruction,
    Field(discriminator="kind"),
]


class PartitionRejected(StrictModel):
    """A source-bound rejection and its first defining obstruction."""

    kind: Literal["NOT_A_PARTITION"] = "NOT_A_PARTITION"
    parts: tuple[PartitionItem, ...] = Field(
        min_length=0,
        max_length=MAX_PARTITION_SIZE,
        description=(
            "The exact bounded JSON-safe candidate sequence classified by this result; "
            "obstruction results are not subject to the partition size cap."
        ),
    )
    obstruction: PartitionObstruction

PartitionCheckBranch = Annotated[
    PartitionFound | PartitionRejected, Field(discriminator="kind")
]


class PartitionCheckResult(StrictModel):
    """Discriminated mathematical outcome: partition or first obstruction."""

    outcome: PartitionCheckBranch


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
