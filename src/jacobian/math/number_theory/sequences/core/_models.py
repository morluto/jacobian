"""Typed wire contracts for finite integer-sequence operations."""

from __future__ import annotations

from pydantic import Field

from jacobian._exact import (
    CanonicalRational,
    ExactInteger,
)
from jacobian._models import StrictModel
from jacobian.math.number_theory.sequences.core.values import (
    MAX_SEQUENCE_LENGTH,
)


class IntegerSequenceValueResult(StrictModel):
    """One canonical integer produced by a sequence aggregate."""

    value: ExactInteger


class IntegerSequenceRationalResult(StrictModel):
    """One reduced rational produced by a sequence statistic."""

    value: CanonicalRational


class IntegerSequenceListResult(StrictModel):
    """A finite list of canonical integers produced by a sequence transform."""

    values: tuple[ExactInteger, ...] = Field(
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

    value: ExactInteger
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


class FiniteIntegerSequence(StrictModel):
    """A possibly empty finite integer sequence."""

    values: tuple[ExactInteger, ...] = Field(
        min_length=0, max_length=MAX_SEQUENCE_LENGTH
    )


class AutocorrelationCell(StrictModel):
    lag: int
    value: ExactInteger


class AutocorrelationResult(StrictModel):
    source: FiniteIntegerSequence
    cells: tuple[AutocorrelationCell, ...] = Field(
        min_length=0, max_length=2 * MAX_SEQUENCE_LENGTH - 1
    )


class SequenceLogConcavityRow(StrictModel):
    index: int = Field(ge=1, le=MAX_SEQUENCE_LENGTH - 2)
    square: ExactInteger
    neighbor_product: ExactInteger
    holds: bool


class SequenceOrderShapeResult(StrictModel):
    source: FiniteIntegerSequence
    first_nondecreasing_violation: int | None = Field(
        default=None, ge=0, le=MAX_SEQUENCE_LENGTH - 2
    )
    first_nonincreasing_violation: int | None = Field(
        default=None, ge=0, le=MAX_SEQUENCE_LENGTH - 2
    )
    weak_unimodal_peak_positions: tuple[int, ...] = Field(
        min_length=0, max_length=MAX_SEQUENCE_LENGTH
    )
    log_concavity_rows: tuple[SequenceLogConcavityRow, ...] = Field(
        min_length=0, max_length=MAX_SEQUENCE_LENGTH - 2
    )
    is_nonnegative: bool
    has_internal_zero: bool
