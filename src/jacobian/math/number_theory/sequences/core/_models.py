"""Typed wire contracts for finite integer-sequence operations."""

from __future__ import annotations

from pydantic import Field

from jacobian._exact import (
    CanonicalInteger,
    CanonicalRational,
)
from jacobian._models import StrictModel
from jacobian.math.number_theory.sequences.core.values import (
    MAX_INTEGER_SEQUENCE_ITEM_DIGITS as MAX_INTEGER_SEQUENCE_ITEM_DIGITS,
)
from jacobian.math.number_theory.sequences.core.values import (
    MAX_SEQUENCE_LENGTH,
    IntegerSequence,
)

# The request is the wire name for the canonical native sequence value.  An
# alias deliberately keeps producer/consumer composition on one Pydantic type.
IntegerSequenceRequest = IntegerSequence


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
        max_length=MAX_SEQUENCE_LENGTH,
    )


class IntegerSequenceIndexListResult(StrictModel):
    """Zero-based indices produced by a sequence search operation."""

    indices: tuple[int, ...] = Field(
        min_length=0,
        max_length=MAX_SEQUENCE_LENGTH,
    )

    def __len__(self) -> int:
        return len(self.indices)


class FrequencyEntry(StrictModel):
    """One distinct sequence value and its positive occurrence count."""

    value: CanonicalInteger
    count: int = Field(ge=1, le=MAX_SEQUENCE_LENGTH)


class IntegerSequenceFrequenciesResult(StrictModel):
    """Frequency entries sorted by ascending value."""

    entries: tuple[FrequencyEntry, ...] = Field(
        min_length=1,
        max_length=MAX_SEQUENCE_LENGTH,
    )


class IntegerSequenceBooleanResult(StrictModel):
    """Truth value of a sequence predicate."""

    holds: bool
