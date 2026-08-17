"""Typed wire contracts for exact algebraic combinatorics operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel

MAX_PARTITION_SIZE = 50
MAX_PARTS = 50


class Partition(StrictModel):
    """One decreasing sequence of positive integers (a Young diagram shape)."""

    parts: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_PARTS)

    @model_validator(mode="after")
    def require_decreasing_positive(self) -> Self:
        if any(part <= 0 for part in self.parts):
            raise ValueError("partition parts must be positive")
        if any(self.parts[i] < self.parts[i + 1] for i in range(len(self.parts) - 1)):
            raise ValueError("partition parts must be non-increasing")
        if sum(self.parts) > MAX_PARTITION_SIZE:
            raise ValueError(f"partition size must not exceed {MAX_PARTITION_SIZE}")
        return self


class HookLengthRequest(StrictModel):
    """Compute the hook lengths of a partition."""

    partition: Partition


class StandardYoungTableauCountRequest(StrictModel):
    """Count standard Young tableaux of a given shape."""

    partition: Partition


class ConjugatePartitionRequest(StrictModel):
    """Compute the conjugate (transpose) partition."""

    partition: Partition


class HookLengthResult(StrictModel):
    """Hook lengths as a flat list of row-indexed values."""

    hooks: tuple[tuple[int, ...], ...] = Field(min_length=1)
    total_product: CanonicalInteger = Field(description="Product of all hook lengths.")
    method: Literal["HOOK_FORMULA"] = "HOOK_FORMULA"


class StandardYoungTableauCountResult(StrictModel):
    """The number of standard Young tableaux of a given shape."""

    count: CanonicalInteger = Field(description="Number of standard Young tableaux.")
    n: int = Field(ge=1, le=MAX_PARTITION_SIZE)
    method: Literal["HOOK_LENGTH_FORMULA"] = "HOOK_LENGTH_FORMULA"


class ConjugatePartitionResult(StrictModel):
    """The conjugate (transpose) partition."""

    conjugate: tuple[int, ...] = Field(min_length=1)
    method: Literal["FERRERS_TRANSPOSE"] = "FERRERS_TRANSPOSE"


__all__ = [
    "ConjugatePartitionRequest",
    "ConjugatePartitionResult",
    "HookLengthRequest",
    "HookLengthResult",
    "Partition",
    "StandardYoungTableauCountRequest",
    "StandardYoungTableauCountResult",
]
