"""Typed wire contracts for finite integer-sequence operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalInteger,
    CanonicalRational,
)
from jacobian._models import StrictModel

_MAX_SEQUENCE_LENGTH = 10_000
MAX_INTEGER_SEQUENCE_ITEM_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS


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
            raise ValueError(
                "sequence item exceeds the "
                f"{MAX_INTEGER_SEQUENCE_ITEM_DIGITS}-digit bound"
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
