"""Typed wire contracts for finite integer-sequence operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalInteger,
    CanonicalRational,
)

_MAX_SEQUENCE_LENGTH = 256
MAX_INTEGER_SEQUENCE_ITEM_DIGITS = MAX_CANONICAL_RATIONAL_DIGITS


class IntegerSequenceRequest(ContractModel):
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


class IntegerSequenceValueResult(ContractModel):
    """One canonical integer produced by a sequence aggregate."""

    value: CanonicalInteger


class IntegerSequenceRationalResult(ContractModel):
    """One reduced rational produced by a sequence statistic."""

    value: CanonicalRational


class IntegerSequenceListResult(ContractModel):
    """A finite list of canonical integers produced by a sequence transform."""

    values: tuple[CanonicalInteger, ...] = Field(
        min_length=0,
        max_length=_MAX_SEQUENCE_LENGTH,
    )


class IntegerSequenceIndexListResult(ContractModel):
    """Zero-based indices produced by a sequence search operation."""

    indices: tuple[int, ...] = Field(
        min_length=0,
        max_length=_MAX_SEQUENCE_LENGTH,
    )

    def __len__(self) -> int:
        return len(self.indices)


class FrequencyEntry(ContractModel):
    """One distinct sequence value and its positive occurrence count."""

    value: CanonicalInteger
    count: int = Field(ge=1, le=_MAX_SEQUENCE_LENGTH)


class IntegerSequenceFrequenciesResult(ContractModel):
    """Frequency entries sorted by ascending value."""

    entries: tuple[FrequencyEntry, ...] = Field(
        min_length=1,
        max_length=_MAX_SEQUENCE_LENGTH,
    )


class IntegerSequenceBooleanResult(ContractModel):
    """Truth value of a sequence predicate."""

    holds: bool
