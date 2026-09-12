"""Typed wire contracts for finite exact-sequence operations."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, ValidationInfo, model_validator
from pydantic_core import PydanticCustomError, core_schema

from jacobian._exact import (
    MAX_CANONICAL_RATIONAL_DIGITS,
    CanonicalRational,
    ExactInteger,
)
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.number_theory.sequences.core.values import (
    MAX_SEQUENCE_LENGTH,
    MAX_SEQUENCE_TOTAL_DIGITS,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"sequences.{reason}", message)


def _rational_sequence_schema(schema: dict[str, Any]) -> None:
    """Publish denominator-one integer wire entries beside rational objects."""

    rational_schema = schema["items"]
    schema["items"] = {
        "oneOf": [
            rational_schema,
            {
                "type": "string",
                "pattern": (
                    rf"^(?:0|-?[1-9][0-9]{{0,{MAX_CANONICAL_RATIONAL_DIGITS - 1}}})"
                    r"(?![\s\S])"
                ),
                "maxLength": MAX_CANONICAL_RATIONAL_DIGITS + 1,
                "description": (
                    "Canonical integer wire entry, interpreted as a "
                    "denominator-one rational."
                ),
            },
        ]
    }


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


class FiniteSequence(StrictModel):
    """Catalog request base for canonical integer or rational sequences."""


class FiniteIntegerSequence(FiniteSequence):
    """A possibly empty finite integer sequence."""

    domain: Literal["integer"] = "integer"
    values: tuple[ExactInteger, ...] = Field(
        min_length=0, max_length=MAX_SEQUENCE_LENGTH
    )

    @model_validator(mode="after")
    def require_bounded_representation(self) -> Self:
        total_digits = sum(
            len(format_canonical_integer(abs(value))) for value in self.values
        )
        if total_digits > MAX_SEQUENCE_TOTAL_DIGITS:
            raise _validation_error(
                "representation_too_large",
                "integer sequence exceeds the "
                f"{MAX_SEQUENCE_TOTAL_DIGITS}-digit representation bound",
            )
        return self


class FiniteRationalSequence(FiniteSequence):
    """A possibly empty finite sequence of canonical real rationals.

    Integer wire entries are accepted as denominator-one rationals.  The
    parsed value is always a canonical rational, so rational consumers do not
    have to infer a coefficient domain from an empty or degenerate sequence.
    """

    domain: Literal["rational"] = "rational"
    values: tuple[CanonicalRational, ...] = Field(
        min_length=0,
        max_length=MAX_SEQUENCE_LENGTH,
        json_schema_extra=_rational_sequence_schema,
    )

    @model_validator(mode="before")
    @classmethod
    def accept_integer_wire_entries(cls, data: object, info: ValidationInfo) -> object:
        if not isinstance(data, dict) or not isinstance(
            data.get("values"), (list, tuple)
        ):
            return data
        converted: list[object] = []
        for value in data["values"]:
            if isinstance(value, int) and not isinstance(value, bool):
                converted.append({"num": value, "den": 1})
                continue
            if isinstance(value, str):
                try:
                    integer = parse_canonical_integer(value)
                except ValueError:
                    pass
                else:
                    if value == format_canonical_integer(integer):
                        if info.mode == "json":
                            converted.append({"num": value, "den": "1"})
                        else:
                            converted.append({"num": integer, "den": 1})
                        continue
            converted.append(value)
        return {**data, "values": tuple(converted)}

    @model_validator(mode="after")
    def require_bounded_representation(self) -> Self:
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
    value: ExactInteger | CanonicalRational = Field(union_mode="left_to_right")


class AutocorrelationResult(StrictModel):
    convention: Literal["aperiodic", "cyclic"]
    source: Annotated[
        FiniteIntegerSequence | FiniteRationalSequence,
        Field(discriminator="domain"),
    ]
    cells: tuple[AutocorrelationCell, ...] = Field(
        min_length=0, max_length=2 * MAX_SEQUENCE_LENGTH - 1
    )

    @model_validator(mode="after")
    def require_canonical_axis_and_domain(self) -> Self:
        size = len(self.source.values)
        expected_lags = (
            range(-(size - 1), size) if self.convention == "aperiodic" else range(size)
        )
        if tuple(cell.lag for cell in self.cells) != tuple(expected_lags):
            raise _validation_error(
                "invalid_lag_axis",
                f"{self.convention} autocorrelation must retain its canonical lag axis",
            )
        integer_source = isinstance(self.source, FiniteIntegerSequence)
        if any(
            isinstance(cell.value, CanonicalRational) is integer_source
            for cell in self.cells
        ):
            raise _validation_error(
                "mixed_coefficient_domain",
                "autocorrelation cells must retain the source coefficient domain",
            )
        return self


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


@classmethod
def _finite_sequence_core_schema(
    cls, source_type: Any, handler: Any
) -> core_schema.CoreSchema:
    if cls is not FiniteSequence:
        return handler(source_type)
    return core_schema.union_schema(
        [
            handler.generate_schema(FiniteRationalSequence),
            handler.generate_schema(FiniteIntegerSequence),
        ]
    )


@classmethod
def _finite_sequence_json_schema(
    cls, core_schema_obj: Any, handler: Any
) -> dict[str, Any]:
    if cls is not FiniteSequence:
        return handler(core_schema_obj)
    schema = dict(handler(core_schema_obj))
    if "type" not in schema:
        schema["type"] = "object"
    return schema


FiniteSequence.__get_pydantic_core_schema__ = _finite_sequence_core_schema
FiniteSequence.__get_pydantic_json_schema__ = _finite_sequence_json_schema
FiniteSequence.model_rebuild(force=True)
