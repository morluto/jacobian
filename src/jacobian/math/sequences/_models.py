"""Typed wire contracts for finite integer-sequence operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalInteger,
    CanonicalRational,
)
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits

_MAX_SEQUENCE_LENGTH = 100_000
MAX_INTEGER_SEQUENCE_ITEM_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS
_MAX_SEQUENCE_WIRE_BYTES = CanonicalLimits().max_output_bytes // 2


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by sequence contracts."""

    return PydanticCustomError(f"sequences.{reason}", message)


class IntegerSequenceRequest(StrictModel):
    """One nonempty finite sequence of canonical integers."""

    values: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=_MAX_SEQUENCE_LENGTH,
    )

    @model_validator(mode="after")
    def require_bounded_items(self) -> Self:
        if any(
            len(value.lstrip("-")) > MAX_INTEGER_SEQUENCE_ITEM_DIGITS
            for value in self.values
        ):
            raise _validation_error(
                "item_too_large",
                "sequence item exceeds the "
                f"{MAX_INTEGER_SEQUENCE_ITEM_DIGITS}-digit bound",
            )
        # Result-sensitive transport guard: estimate wire bytes before allocation.
        # Small 1-digit values at 100k ~400KB pass; 10k x 256-digit ~2.5MB pass;
        # 400k x 256-digit would exceed 10MiB and is rejected here with a
        # composition hint rather than a coarse n ceiling.
        estimated = sum(len(value) + 3 for value in self.values) + 64
        if estimated > _MAX_SEQUENCE_WIRE_BYTES:
            raise _validation_error(
                "transport_too_large",
                "sequence request exceeds the "
                f"{_MAX_SEQUENCE_WIRE_BYTES}-byte transport envelope; "
                "chunk the sequence into ≤10MiB pieces and compose via typed values",
            )
        return self


class IntegerSequenceValueResult(StrictModel):
    """One canonical integer produced by a sequence aggregate."""

    value: CanonicalInteger


class IntegerSequenceRationalResult(StrictModel):
    """One reduced rational produced by a sequence statistic."""

    value: CanonicalRational


class IntegerSequenceListResult(StrictModel):
    """A finite list of canonical integers produced by a sequence transform."""

    values: tuple[CanonicalInteger, ...] = Field(
        min_length=0,
        max_length=_MAX_SEQUENCE_LENGTH,
    )


class IntegerSequenceIndexListResult(StrictModel):
    """Zero-based indices produced by a sequence search operation."""

    indices: tuple[int, ...] = Field(
        min_length=0,
        max_length=_MAX_SEQUENCE_LENGTH,
    )

    def __len__(self) -> int:
        return len(self.indices)


class FrequencyEntry(StrictModel):
    """One distinct sequence value and its positive occurrence count."""

    value: CanonicalInteger
    count: int = Field(ge=1, le=_MAX_SEQUENCE_LENGTH)


class IntegerSequenceFrequenciesResult(StrictModel):
    """Frequency entries sorted by ascending value."""

    entries: tuple[FrequencyEntry, ...] = Field(
        min_length=1,
        max_length=_MAX_SEQUENCE_LENGTH,
    )


class IntegerSequenceBooleanResult(StrictModel):
    """Truth value of a sequence predicate."""

    holds: bool
