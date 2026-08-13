"""Typed wire contracts for finite integer-sequence operations."""

from __future__ import annotations

from pydantic import Field

from jacobian.contracts.exact import CanonicalInteger, CanonicalRational
from jacobian.contracts.results import ContractModel

_MAX_SEQUENCE_LENGTH = 256


class IntegerSequenceRequest(ContractModel):
    """One nonempty finite sequence of canonical integers."""

    values: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=_MAX_SEQUENCE_LENGTH,
    )


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
