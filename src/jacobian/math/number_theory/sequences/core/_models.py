"""Typed wire contracts for finite sequence operations."""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import (
    CanonicalRational,
    ExactInteger,
)
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.number_theory.sequences.core.values import (
    MAX_SEQUENCE_LENGTH,
    MAX_SEQUENCE_TOTAL_DIGITS,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"sequences.{reason}", message)


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


class FiniteRationalSequence(StrictModel):
    """A possibly empty finite sequence of canonical rational values.

    Integer wire entries are accepted as denominator-one rationals, but the
    parsed value always has this one rational carrier.  Keeping the carrier
    separate from ``FiniteIntegerSequence`` prevents an integer-only consumer
    from silently receiving a rational sequence.
    """

    values: tuple[CanonicalRational, ...] = Field(
        min_length=0, max_length=MAX_SEQUENCE_LENGTH
    )

    @model_validator(mode="before")
    @classmethod
    def accept_integer_wire_entries(cls, data: object) -> object:
        """Normalize canonical integer JSON to denominator-one rationals."""

        if not isinstance(data, dict) or not isinstance(
            data.get("values"), (list, tuple)
        ):
            return data
        converted: list[object] = []
        for value in data["values"]:
            if isinstance(value, int) and not isinstance(value, bool):
                converted.append({"num": format_canonical_integer(value), "den": "1"})
                continue
            if isinstance(value, str):
                try:
                    integer = int(value)
                except ValueError:
                    pass
                else:
                    if value == str(integer):
                        converted.append({"num": value, "den": "1"})
                        continue
            converted.append(value)
        return {**data, "values": tuple(converted)}

    @model_validator(mode="after")
    def require_bounded_representation(self) -> FiniteRationalSequence:
        total_digits = sum(
            len(format_canonical_integer(abs(value.num)))
            + len(format_canonical_integer(value.den))
            for value in self.values
        )
        if total_digits > MAX_SEQUENCE_TOTAL_DIGITS:
            raise _validation_error(
                "representation_too_large",
                "rational sequence exceeds the "
                f"{MAX_SEQUENCE_TOTAL_DIGITS}-digit representation bound",
            )
        return self


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
    square: ExactInteger | CanonicalRational
    neighbor_product: ExactInteger | CanonicalRational
    holds: bool


class SequenceOrderShapeResult(StrictModel):
    source: FiniteIntegerSequence | FiniteRationalSequence
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
    first_log_concavity_violation: int | None = Field(
        default=None, ge=1, le=MAX_SEQUENCE_LENGTH - 2
    )
    is_nonnegative: bool
    first_negative_index: int | None = Field(
        default=None, ge=0, le=MAX_SEQUENCE_LENGTH - 1
    )
    has_internal_zero: bool
    first_internal_zero_index: int | None = Field(
        default=None, ge=1, le=MAX_SEQUENCE_LENGTH - 2
    )

    @model_validator(mode="after")
    def require_structural_profile(self) -> SequenceOrderShapeResult:
        """Check profile shape without replaying any arithmetic invariant."""

        size = len(self.source.values)
        peaks = self.weak_unimodal_peak_positions
        if peaks != tuple(sorted(set(peaks))) or any(
            index < 0 or index >= size for index in peaks
        ):
            raise ValueError(
                "unimodal peak positions must be sorted, unique, and in range"
            )
        expected_rows = tuple(range(1, max(size - 1, 1)))
        row_indices = tuple(row.index for row in self.log_concavity_rows)
        if row_indices != expected_rows:
            raise ValueError(
                "log-concavity rows must cover each interior index exactly once"
            )
        if self.first_log_concavity_violation is not None and (
            self.first_log_concavity_violation not in row_indices
        ):
            raise ValueError("log-concavity violation must identify an interior row")
        if self.first_negative_index is not None and self.first_negative_index >= size:
            raise ValueError("negative index must identify a source position")
        if self.first_internal_zero_index is not None and (
            self.first_internal_zero_index not in row_indices
        ):
            raise ValueError("internal-zero index must identify an interior position")
        return self
