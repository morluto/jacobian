"""Typed wire contracts for exact bounded impartial-game operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.games.impartial.values import (
    MAX_HEAP_BOUND,
    MAX_HEAP_SIZE,
    MAX_HEAPS,
    MAX_NIM_DISTINCT_OPTIONS,
    MAX_NIM_RAW_CANDIDATES,
    MAX_POSITIONS,
    MAX_SUBTRACTION_VALUE,
    MAX_SUBTRACTION_WORK,
    ImpartialGame,
    NimOption,
    NimPosition,
)

MAX_COMPONENT_GRUNDY = MAX_POSITIONS - 1
MAX_DISJUNCTIVE_GRUNDY = (1 << MAX_COMPONENT_GRUNDY.bit_length()) - 1


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
    def require_bounded_complete_shape(self) -> Self:
        values = tuple(entry.grundy for entry in self.entries)
        expected_max = max(values, default=0)
        expected_histogram = tuple(
            values.count(index) for index in range(expected_max + 1)
        )
        if (
            tuple(entry.position for entry in self.entries) != self.game.positions
            or any(
                entry.option_grundy_set != tuple(sorted(set(entry.option_grundy_set)))
                for entry in self.entries
            )
            or any(value > MAX_POSITIONS - 1 for value in values)
            or self.max_grundy != expected_max
            or self.histogram != expected_histogram
            or set(self.topological_order) != set(self.game.positions)
            or len(self.topological_order) != len(self.game.positions)
        ):
            raise PydanticCustomError(
                "impartial_game.grundy_table_shape",
                "result must have a bounded canonical complete-table shape",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: GrundyTableRequest,
        entries: tuple[GrundyEntry, ...],
        max_grundy: int,
        histogram: tuple[int, ...],
        topological_order: tuple[str, ...],
    ) -> Self:
        return cls(
            game=request.game,
            entries=entries,
            max_grundy=max_grundy,
            histogram=histogram,
            topological_order=topological_order,
        )


class BirthdayRequest(StrictModel):
    game: ImpartialGame


class BirthdayResult(BirthdayRequest):
    birthdays: tuple[tuple[str, int], ...]
    complete: Literal[True] = True
    method: Literal["REVERSE_TOPOLOGICAL_HEIGHT"] = "REVERSE_TOPOLOGICAL_HEIGHT"

    @model_validator(mode="after")
    def require_bounded_complete_shape(self) -> Self:
        if tuple(
            position for position, _ in self.birthdays
        ) != self.game.positions or any(
            birthday < 0 or birthday >= len(self.game.positions)
            for _, birthday in self.birthdays
        ):
            raise PydanticCustomError(
                "impartial_game.birthday_table_shape",
                "result must have a bounded canonical complete-table shape",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, request: BirthdayRequest, birthdays: tuple[tuple[str, int], ...]
    ) -> Self:
        return cls(game=request.game, birthdays=birthdays)


class SubtractionGrundyPrefixRequest(StrictModel):
    subtraction_set: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_SUBTRACTION_VALUE
    )
    max_heap: int = Field(ge=0, le=MAX_HEAP_BOUND)

    @model_validator(mode="after")
    def require_canonical_bounded_input(self) -> Self:
        if self.subtraction_set != tuple(sorted(set(self.subtraction_set))):
            raise PydanticCustomError(
                "impartial_game.subtraction_set_not_canonical",
                "subtraction set must be distinct and sorted",
            )
        if any(
            not 1 <= value <= MAX_SUBTRACTION_VALUE for value in self.subtraction_set
        ):
            raise PydanticCustomError(
                "impartial_game.subtraction_value_out_of_bounds",
                "subtraction value is outside the supported bound",
            )
        if len(self.subtraction_set) * (self.max_heap + 1) > MAX_SUBTRACTION_WORK:
            raise PydanticCustomError(
                "impartial_game.subtraction_work_exceeded",
                "subtraction Grundy computation exceeds the work bound",
            )
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
    def require_bounded_complete_shape(self) -> Self:
        expected_p = tuple(
            heap for heap, value in enumerate(self.grundy_values) if value == 0
        )
        expected_n = tuple(
            heap for heap, value in enumerate(self.grundy_values) if value != 0
        )
        if (
            len(self.grundy_values) != self.max_heap + 1
            or len(self.option_sets) != self.max_heap + 1
            or any(
                value < 0 or value > len(self.subtraction_set)
                for value in self.grundy_values
            )
            or any(
                options != tuple(sorted(set(options))) for options in self.option_sets
            )
            or self.p_positions != expected_p
            or self.n_positions != expected_n
        ):
            raise PydanticCustomError(
                "impartial_game.grundy_prefix_shape",
                "result must have a bounded canonical complete-prefix shape",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: SubtractionGrundyPrefixRequest,
        grundy_values: tuple[int, ...],
        option_sets: tuple[tuple[int, ...], ...],
        p_positions: tuple[int, ...],
        n_positions: tuple[int, ...],
    ) -> Self:
        return cls(
            subtraction_set=request.subtraction_set,
            max_heap=request.max_heap,
            grundy_values=grundy_values,
            option_sets=option_sets,
            p_positions=p_positions,
            n_positions=n_positions,
        )


__all__ = [
    "BirthdayRequest",
    "BirthdayResult",
    "GrundyEntry",
    "GrundyTableRequest",
    "GrundyTableResult",
    "NimOptionsRequest",
    "NimOptionsResult",
    "SubtractionGrundyPrefixRequest",
    "SubtractionGrundyPrefixResult",
]


# ---------------------------------------------------------------------------
# Nim sum operations
# ---------------------------------------------------------------------------


class NimSumRequest(StrictModel):
    """One canonical finite normal-play Nim position."""

    position: NimPosition


class NimSumResult(NimSumRequest):
    """The exact bitwise xor bound to its canonical source position."""

    nim_sum: StrictInt = Field(ge=0, le=(1 << MAX_HEAP_SIZE.bit_length()) - 1)
    is_p_position: StrictBool

    @model_validator(mode="after")
    def require_status_semantics(self) -> Self:
        if self.is_p_position != (self.nim_sum == 0):
            raise PydanticCustomError(
                "impartial_game.p_position_mismatch",
                "is_p_position must report whether the exact xor is zero",
            )
        return self

    @classmethod
    def _from_kernel(cls, request: NimSumRequest, nim_sum: int) -> Self:
        return cls(
            position=request.position, nim_sum=nim_sum, is_p_position=(nim_sum == 0)
        )


class NimOptionsRequest(StrictModel):
    """Enumerate the complete distinct option family of one Nim multiset."""

    position: NimPosition = Field(
        description=(
            "Canonical sorted heap multiset whose one-move option family is requested."
        )
    )


class NimOptionsResult(NimOptionsRequest):
    """Every distinct legal one-heap reduction, in option-position order."""

    options: tuple[NimOption, ...] = Field(
        max_length=MAX_NIM_DISTINCT_OPTIONS,
        description="Distinct resulting positions in lexicographic heap order.",
    )
    raw_candidate_count: int = Field(
        ge=0,
        le=MAX_NIM_RAW_CANDIDATES,
        description=(
            "Number of indexed (source heap, replacement size) moves before "
            "canonical multiset deduplication."
        ),
    )
    distinct_option_count: int = Field(
        ge=0,
        le=MAX_NIM_DISTINCT_OPTIONS,
        description="Number of distinct canonical resulting positions.",
    )
    complete: Literal[True] = True

    @model_validator(mode="after")
    def require_bounded_complete_shape(self) -> Self:
        if (
            self.distinct_option_count != len(self.options)
            or tuple(option.resulting_position.heaps for option in self.options)
            != tuple(sorted(option.resulting_position.heaps for option in self.options))
            or len({option.resulting_position.heaps for option in self.options})
            != len(self.options)
        ):
            raise PydanticCustomError(
                "impartial_game.nim_options_shape",
                "result must have a bounded canonical Nim-option shape",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: NimOptionsRequest,
        options: tuple[NimOption, ...],
        raw_candidate_count: int,
        distinct_option_count: int,
    ) -> Self:
        return cls(
            position=request.position,
            options=options,
            raw_candidate_count=raw_candidate_count,
            distinct_option_count=distinct_option_count,
        )


class OutcomeProfileRequest(StrictModel):
    """Request the P/N outcome partition of an impartial game."""

    game: ImpartialGame


class OutcomeProfileResult(StrictModel):
    """The complete P/N position partition with Grundy values."""

    p_positions: tuple[str, ...]
    n_positions: tuple[str, ...]
    grundy_values: tuple[tuple[str, int], ...]
    terminal_positions: tuple[str, ...]


# ---------------------------------------------------------------------------
# Disjunctive sum operation
# ---------------------------------------------------------------------------


class DisjunctiveSumRequest(StrictModel):
    """A disjunctive sum of finite impartial game components.

    Each component specifies a game DAG and the label of the starting
    position whose Grundy value represents that component in the sum.
    The Grundy value of the disjunctive sum is the bitwise XOR of the
    component Grundy values.
    """

    components: tuple[ImpartialGame, ...] = Field(min_length=1, max_length=MAX_HEAPS)
    start_positions: tuple[str, ...] = Field(min_length=1, max_length=MAX_HEAPS)

    @model_validator(mode="after")
    def require_matching_bounded(self) -> Self:
        if len(self.components) != len(self.start_positions):
            raise PydanticCustomError(
                "impartial_game.component_count_mismatch",
                "components and start_positions must have equal length",
            )
        for index, (game, start) in enumerate(
            zip(self.components, self.start_positions, strict=True)
        ):
            if start not in game.positions:
                raise PydanticCustomError(
                    "impartial_game.start_position_unknown",
                    f"start position {index!r} is not in component {index}'s positions",
                )
        return self


class DisjunctiveSumResult(StrictModel):
    """The exact Grundy value of a disjunctive sum of impartial games."""

    grundy_value: int = Field(ge=0, le=MAX_DISJUNCTIVE_GRUNDY)
    component_grundy_values: tuple[int, ...] = Field(
        min_length=1,
        max_length=MAX_HEAPS,
    )
    is_p_position: bool
    component_count: int = Field(ge=1, le=MAX_HEAPS)

    @model_validator(mode="after")
    def require_exact_disjunctive_invariants(self) -> Self:
        if self.component_count != len(self.component_grundy_values):
            raise PydanticCustomError(
                "impartial_game.component_count_mismatch",
                "component_count must match component_grundy_values length",
            )
        if any(
            not 0 <= value <= MAX_COMPONENT_GRUNDY
            for value in self.component_grundy_values
        ):
            raise PydanticCustomError(
                "impartial_game.component_grundy_out_of_bounds",
                f"component Grundy values must be between 0 and {MAX_COMPONENT_GRUNDY}",
            )
        from functools import reduce
        from operator import xor

        expected = reduce(xor, self.component_grundy_values, 0)
        if self.grundy_value != expected:
            raise PydanticCustomError(
                "impartial_game.grundy_value_mismatch",
                "grundy_value must be XOR of component_grundy_values",
            )
        if self.is_p_position != (expected == 0):
            raise PydanticCustomError(
                "impartial_game.p_position_mismatch",
                "is_p_position must agree with grundy_value == 0",
            )
        return self
