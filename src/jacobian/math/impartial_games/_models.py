"""Typed wire contracts for exact bounded impartial-game operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.impartial_games.operations import (
    birthdays,
    grundy_table,
    subtraction_grundy_prefix,
)
from jacobian.math.impartial_games.values import (
    MAX_HEAP_BOUND,
    MAX_HEAP_SIZE,
    MAX_HEAPS,
    MAX_SUBTRACTION_VALUE,
    MAX_SUBTRACTION_WORK,
    ImpartialGame,
)


class GrundyTableRequest(StrictModel):
    game: ImpartialGame


class GrundyEntry(StrictModel):
    position: str
    grundy: int = Field(ge=0)
    option_grundy_set: tuple[int, ...]


class GrundyTableResult(GrundyTableRequest):
    entries: tuple[GrundyEntry, ...]
    max_grundy: int = Field(ge=0)
    histogram: tuple[int, ...]
    topological_order: tuple[str, ...]
    complete: Literal[True] = True
    method: Literal["REVERSE_TOPOLOGICAL_MEX"] = "REVERSE_TOPOLOGICAL_MEX"

    @model_validator(mode="after")
    def bind_complete_table(self) -> Self:
        analysis = grundy_table(self.game)
        option_sets = dict(analysis.option_value_sets)
        expected_entries = tuple(
            GrundyEntry(
                position=position,
                grundy=value,
                option_grundy_set=option_sets[position],
            )
            for position, value in analysis.values
        )
        values = tuple(value for _, value in analysis.values)
        expected_max = max(values, default=0)
        expected_histogram = tuple(
            values.count(index) for index in range(expected_max + 1)
        )
        if (
            self.entries != expected_entries
            or self.max_grundy != expected_max
            or self.histogram != expected_histogram
            or self.topological_order != analysis.topological_order
        ):
            raise ValueError("result must be the exact complete Grundy table")
        return self


class BirthdayRequest(StrictModel):
    game: ImpartialGame


class BirthdayResult(BirthdayRequest):
    birthdays: tuple[tuple[str, int], ...]
    complete: Literal[True] = True
    method: Literal["REVERSE_TOPOLOGICAL_HEIGHT"] = "REVERSE_TOPOLOGICAL_HEIGHT"

    @model_validator(mode="after")
    def bind_birthdays(self) -> Self:
        if self.birthdays != birthdays(self.game):
            raise ValueError("result must be the exact complete birthday table")
        return self


class SubtractionGrundyPrefixRequest(StrictModel):
    subtraction_set: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_SUBTRACTION_VALUE
    )
    max_heap: int = Field(ge=0, le=MAX_HEAP_BOUND)

    @model_validator(mode="after")
    def require_canonical_bounded_input(self) -> Self:
        if self.subtraction_set != tuple(sorted(set(self.subtraction_set))):
            raise ValueError("subtraction set must be distinct and sorted")
        if any(
            not 1 <= value <= MAX_SUBTRACTION_VALUE for value in self.subtraction_set
        ):
            raise ValueError("subtraction value is outside the supported bound")
        if len(self.subtraction_set) * (self.max_heap + 1) > MAX_SUBTRACTION_WORK:
            raise ValueError("subtraction Grundy computation exceeds the work bound")
        return self


class SubtractionGrundyPrefixResult(SubtractionGrundyPrefixRequest):
    grundy_values: tuple[int, ...]
    option_sets: tuple[tuple[int, ...], ...]
    p_positions: tuple[int, ...]
    n_positions: tuple[int, ...]
    complete: Literal[True] = True
    scope: Literal["HEAPS_ZERO_THROUGH_MAX_HEAP"] = "HEAPS_ZERO_THROUGH_MAX_HEAP"
    method: Literal["BOUNDED_DYNAMIC_PROGRAMMING"] = "BOUNDED_DYNAMIC_PROGRAMMING"

    @model_validator(mode="after")
    def bind_complete_prefix(self) -> Self:
        analysis = subtraction_grundy_prefix(self.subtraction_set, self.max_heap)
        expected_p = tuple(
            heap for heap, value in enumerate(analysis.grundy_values) if value == 0
        )
        expected_n = tuple(
            heap for heap, value in enumerate(analysis.grundy_values) if value != 0
        )
        if (
            self.grundy_values != analysis.grundy_values
            or self.option_sets != analysis.option_value_sets
            or self.p_positions != expected_p
            or self.n_positions != expected_n
        ):
            raise ValueError("result must be the exact complete bounded Grundy prefix")
        return self


__all__ = [
    "BirthdayRequest",
    "BirthdayResult",
    "GrundyEntry",
    "GrundyTableRequest",
    "GrundyTableResult",
    "SubtractionGrundyPrefixRequest",
    "SubtractionGrundyPrefixResult",
]


# ---------------------------------------------------------------------------
# Nim sum operations
# ---------------------------------------------------------------------------


class NimSumRequest(StrictModel):
    """A finite Nim position: a bounded list of nonnegative heap sizes."""

    heaps: tuple[int, ...] = Field(min_length=0, max_length=MAX_HEAPS)

    @model_validator(mode="after")
    def require_bounded_heaps(self) -> Self:
        if any(heap < 0 for heap in self.heaps):
            raise ValueError("heap sizes must be nonnegative")
        if any(heap > MAX_HEAP_SIZE for heap in self.heaps):
            raise ValueError(f"heap sizes must be at most {MAX_HEAP_SIZE}")
        return self


class NimSumResult(StrictModel):
    """The exact nim sum (bitwise xor) of a Nim position."""

    nim_sum: int = Field(ge=0)
    is_p_position: bool
    heaps: tuple[int, ...]


class OutcomeProfileRequest(StrictModel):
    """Request the P/N outcome partition of an impartial game."""

    game: ImpartialGame


class OutcomeProfileResult(StrictModel):
    """The complete P/N position partition with Grundy values."""

    p_positions: tuple[str, ...]
    n_positions: tuple[str, ...]
    grundy_values: tuple[tuple[str, int], ...]
    terminal_positions: tuple[str, ...]
