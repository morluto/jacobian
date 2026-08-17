"""Typed wire contracts for impartial combinatorial game operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_POSITIONS = 500
MAX_MOVES = 2_000
MAX_LABEL_LEN = 64
MAX_HEAPS = 50
MAX_HEAP_SIZE = 10_000
MAX_SUBTRACTION_VALUE = 500
MAX_HEAP_BOUND = 5_000


class GameMove(StrictModel):
    """One directed edge ``source -> target`` in the game DAG."""

    source: str = Field(min_length=1, max_length=MAX_LABEL_LEN)
    target: str = Field(min_length=1, max_length=MAX_LABEL_LEN)


class ImpartialGameDAGRequest(StrictModel):
    """A finite impartial game specified as a complete move relation."""

    positions: tuple[str, ...] = Field(min_length=1, max_length=MAX_POSITIONS)
    moves: tuple[GameMove, ...] = Field(max_length=MAX_MOVES)

    @model_validator(mode="after")
    def validate_dag(self) -> Self:
        labels = set(self.positions)
        if len(labels) != len(self.positions):
            raise ValueError("position labels must be unique")
        seen: set[tuple[str, str]] = set()
        for move in self.moves:
            if move.source not in labels:
                raise ValueError(
                    f"move source {move.source!r} is not a declared position"
                )
            if move.target not in labels:
                raise ValueError(
                    f"move target {move.target!r} is not a declared position"
                )
            if move.source == move.target:
                raise ValueError("self-loops are not allowed")
            key = (move.source, move.target)
            if key in seen:
                raise ValueError(f"duplicate move {key}")
            seen.add(key)
        return self


class GrundyTableRequest(StrictModel):
    """Compute the Grundy table of a finite impartial game DAG."""

    game: ImpartialGameDAGRequest


class PositionGrundyRequest(StrictModel):
    """Compute the Grundy value of a single position."""

    game: ImpartialGameDAGRequest
    position: str = Field(min_length=1, max_length=MAX_LABEL_LEN)

    @model_validator(mode="after")
    def validate_position(self) -> Self:
        if self.position not in set(self.game.positions):
            raise ValueError(f"position {self.position!r} is not in the game")
        return self


class OutcomeProfileRequest(StrictModel):
    """Compute the P/N partition of a finite impartial game."""

    game: ImpartialGameDAGRequest


class NimEquivalentRequest(StrictModel):
    """Find the canonical Nim heap equivalent to one position."""

    game: ImpartialGameDAGRequest
    position: str = Field(min_length=1, max_length=MAX_LABEL_LEN)

    @model_validator(mode="after")
    def validate_position(self) -> Self:
        if self.position not in set(self.game.positions):
            raise ValueError(f"position {self.position!r} is not in the game")
        return self


class GrundyClassesRequest(StrictModel):
    """Partition positions by equal Grundy value."""

    game: ImpartialGameDAGRequest


class BirthdayRequest(StrictModel):
    """Compute birthdays (DAG heights) of all positions."""

    game: ImpartialGameDAGRequest


# -- Results ----------------------------------------------------------------


class GrundyEntry(StrictModel):
    """One position with its Grundy value and option Grundy set."""

    position: str
    grundy: int = Field(ge=0)
    option_grundy_set: tuple[int, ...] = Field(default_factory=tuple)


class GrundyTableResult(StrictModel):
    """Complete Grundy table of a finite impartial game."""

    entry_map: tuple[GrundyEntry, ...]
    max_grundy: int = Field(ge=0)
    histogram: tuple[int, ...]
    topological_order: tuple[str, ...]


class PositionGrundyResult(StrictModel):
    """Grundy value of a single position with its dependency sub-DAG."""

    position: str
    grundy: int = Field(ge=0)
    reachable_positions: tuple[str, ...]
    topological_order: tuple[str, ...]
    option_grundy_set: tuple[int, ...] = Field(default_factory=tuple)


class OutcomeProfileResult(StrictModel):
    """P/N partition of a finite impartial game."""

    p_positions: tuple[str, ...]
    n_positions: tuple[str, ...]
    terminal_positions: tuple[str, ...]
    grundy_map: tuple[tuple[str, int], ...]


class NimEquivalentResult(StrictModel):
    """Canonical Nim heap equivalent to one position."""

    position: str
    heap_size: int = Field(ge=0)


class GrundyClassEntry(StrictModel):
    """One equivalence class of positions with the same Grundy value."""

    grundy: int = Field(ge=0)
    positions: tuple[str, ...]


class GrundyClassesResult(StrictModel):
    """Partition of positions by equal Grundy value."""

    classes: tuple[GrundyClassEntry, ...]
    histogram: tuple[int, ...]


class BirthdayResult(StrictModel):
    """Birthday (DAG height) of every position."""

    birthdays: tuple[tuple[str, int], ...]


# -- Nim --------------------------------------------------------------------


class NimSumRequest(StrictModel):
    """Compute the nim-sum (bitwise xor) of heap sizes."""

    heaps: tuple[int, ...] = Field(min_length=1, max_length=MAX_HEAPS)

    @model_validator(mode="after")
    def validate_heaps(self) -> Self:
        for heap in self.heaps:
            if heap < 0:
                raise ValueError("heap sizes must be non-negative")
            if heap > MAX_HEAP_SIZE:
                raise ValueError(f"heap sizes must be at most {MAX_HEAP_SIZE}")
        return self


class NimSumResult(StrictModel):
    """Result of a nim-sum computation."""

    heaps: tuple[int, ...]
    nim_sum: int = Field(ge=0)
    is_p_position: bool


class NimOptionsRequest(StrictModel):
    """Enumerate all legal options of a Nim position."""

    heaps: tuple[int, ...] = Field(min_length=1, max_length=MAX_HEAPS)

    @model_validator(mode="after")
    def validate_heaps(self) -> Self:
        for heap in self.heaps:
            if heap < 0:
                raise ValueError("heap sizes must be non-negative")
            if heap > MAX_HEAP_SIZE:
                raise ValueError(f"heap sizes must be at most {MAX_HEAP_SIZE}")
        return self


class NimOption(StrictModel):
    """One legal move in Nim: reduce one heap."""

    heap_index: int = Field(ge=0)
    old_size: int = Field(ge=0)
    new_size: int = Field(ge=0)
    resulting_heaps: tuple[int, ...]


class NimOptionsResult(StrictModel):
    """All legal options of a Nim position."""

    options: tuple[NimOption, ...]


# -- Subtraction games -----------------------------------------------------


class SubtractionDAGRequest(StrictModel):
    """Build the game DAG for a bounded subtraction game."""

    subtraction_set: tuple[int, ...] = Field(min_length=1)
    max_heap: int = Field(ge=0, le=MAX_HEAP_BOUND)

    @model_validator(mode="after")
    def validate_subtraction_set(self) -> Self:
        seen: set[int] = set()
        for value in self.subtraction_set:
            if value <= 0:
                raise ValueError("subtraction set values must be positive")
            if value > MAX_SUBTRACTION_VALUE:
                raise ValueError(
                    f"subtraction set values must be at most {MAX_SUBTRACTION_VALUE}"
                )
            if value in seen:
                raise ValueError("subtraction set values must be unique")
            seen.add(value)
        return self


class SubtractionDAGResult(StrictModel):
    """Game DAG for a bounded subtraction game."""

    positions: tuple[str, ...]
    moves: tuple[GameMove, ...]
    terminal_positions: tuple[str, ...]


class SubtractionGrundyPrefixRequest(StrictModel):
    """Compute the Grundy prefix of a bounded subtraction game."""

    subtraction_set: tuple[int, ...] = Field(min_length=1)
    max_heap: int = Field(ge=0, le=MAX_HEAP_BOUND)

    @model_validator(mode="after")
    def validate_subtraction_set(self) -> Self:
        seen: set[int] = set()
        for value in self.subtraction_set:
            if value <= 0:
                raise ValueError("subtraction set values must be positive")
            if value > MAX_SUBTRACTION_VALUE:
                raise ValueError(
                    f"subtraction set values must be at most {MAX_SUBTRACTION_VALUE}"
                )
            if value in seen:
                raise ValueError("subtraction set values must be unique")
            seen.add(value)
        return self


class SubtractionGrundyPrefixResult(StrictModel):
    """Grundy prefix of a bounded subtraction game."""

    grundy_values: tuple[int, ...]
    option_sets: tuple[tuple[int, ...], ...]
    p_positions: tuple[int, ...]
    n_positions: tuple[int, ...]


# -- Mex --------------------------------------------------------------------


class MexRequest(StrictModel):
    """Compute the mex (minimum excluded value) of a bounded finite set."""

    values: tuple[int, ...] = Field(max_length=MAX_POSITIONS)

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        for value in self.values:
            if value < 0:
                raise ValueError("mex values must be non-negative")
        return self


class MexResult(StrictModel):
    """Result of a mex computation."""

    mex: int = Field(ge=0)
    membership_prefix: tuple[int, ...]
    first_gap: int = Field(ge=0)


__all__ = [
    "BirthdayRequest",
    "BirthdayResult",
    "GameMove",
    "GrundyClassEntry",
    "GrundyClassesRequest",
    "GrundyClassesResult",
    "GrundyEntry",
    "GrundyTableRequest",
    "GrundyTableResult",
    "ImpartialGameDAGRequest",
    "MexRequest",
    "MexResult",
    "NimEquivalentRequest",
    "NimEquivalentResult",
    "NimOption",
    "NimOptionsRequest",
    "NimOptionsResult",
    "NimSumRequest",
    "NimSumResult",
    "OutcomeProfileRequest",
    "OutcomeProfileResult",
    "PositionGrundyRequest",
    "PositionGrundyResult",
    "SubtractionDAGRequest",
    "SubtractionDAGResult",
    "SubtractionGrundyPrefixRequest",
    "SubtractionGrundyPrefixResult",
]
