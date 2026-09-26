"""Canonical values for bounded RSK correspondences."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.symmetric_functions.values import (
    MAX_PARTITION_SIZE,
    IntegerPartition,
    SemistandardYoungTableau,
    StandardYoungTableau,
)
from jacobian.math.logic.languages.words.values import (
    MAX_ALPHABET_SIZE,
    MAX_SYMBOL_LENGTH,
    Symbol,
)

# An N-letter word produces two N-cell tableaux, so the canonical tableau
# cell bound derives the word-length envelope.
MAX_RSK_WORD_LENGTH = MAX_PARTITION_SIZE
# Every insertion or reverse-insertion step binary-searches one row of at
# most MAX_RSK_WORD_LENGTH entries.
MAX_RSK_ROW_SEARCH_COMPARISONS = MAX_RSK_WORD_LENGTH.bit_length()
# A word at the length boundary over an alphabet of MAX_ALPHABET_SIZE symbols
# carries at most MAX_SYMBOL_LENGTH Unicode scalar values per symbol, so the
# retained alphabet-plus-letters payload is bounded by a symbol cardinality.
MAX_RSK_WORD_PAYLOAD_SCALARS = (
    MAX_RSK_WORD_LENGTH + MAX_ALPHABET_SIZE
) * MAX_SYMBOL_LENGTH
# A row reading retains the alphabet once and emits one alphabet symbol per
# tableau cell, so its payload is bounded by the same word cardinality.
MAX_RSK_ALPHABET_RANK_DIGITS = len(str(MAX_ALPHABET_SIZE))
RSKConvention = Literal["ROW_INSERTION_RSK_V1"]


class RSKTableauPair(StrictModel):
    __doc__ = f"""The compact ordinary-word RSK image under row insertion.

    Insertion-tableau entries are one-based ranks in ``alphabet``.  The
    alphabet therefore remains attached to the pair and makes inverse RSK an
    exact operation even when symbols are not integers.  The common shape has
    at most {MAX_PARTITION_SIZE} cells, the canonical partition-size bound.
    """

    alphabet: tuple[Symbol, ...] = Field(
        max_length=MAX_ALPHABET_SIZE,
        description=(
            "The exact ordered source alphabet; insertion-tableau entry i "
            "denotes alphabet[i - 1]. The empty alphabet is canonical exactly "
            "when both tableaux and their common shape are empty."
        ),
        json_schema_extra={"uniqueItems": True},
    )
    insertion_tableau: SemistandardYoungTableau
    recording_tableau: StandardYoungTableau
    shape: IntegerPartition = Field(
        description=(
            "The common tableau shape, required to equal both derived row-length "
            f"partitions and to contain at most {MAX_PARTITION_SIZE} cells."
        )
    )
    source_kind: Literal["WORD"] = "WORD"
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"

    @model_validator(mode="after")
    def require_compatible_word_pair(self) -> Self:
        if len(set(self.alphabet)) != len(self.alphabet):
            raise PydanticCustomError(
                "algebraic_combinatorics.alphabet_not_unique",
                "alphabet symbols must be distinct",
            )
        if self.insertion_tableau.shape != self.shape:
            raise PydanticCustomError(
                "algebraic_combinatorics.insertion_shape_mismatch",
                "insertion tableau shape must equal the common shape",
            )
        if self.recording_tableau.shape != self.shape:
            raise PydanticCustomError(
                "algebraic_combinatorics.recording_shape_mismatch",
                "recording tableau shape must equal the common shape",
            )
        return self


class RSKBumpStep(StrictModel):
    """One row replacement during ordinary word row insertion."""

    row: StrictInt = Field(ge=0, le=MAX_RSK_WORD_LENGTH - 1)
    column: StrictInt = Field(ge=0, le=MAX_RSK_WORD_LENGTH - 1)
    bumped_entry: StrictInt = Field(ge=1, le=MAX_ALPHABET_SIZE)


class RSKInsertionEvent(StrictModel):
    """One source letter's bump path and new terminal cell, with zero-based cells."""

    position: StrictInt = Field(ge=1, le=MAX_RSK_WORD_LENGTH)
    letter: Symbol
    bump_path: tuple[RSKBumpStep, ...] = Field(max_length=MAX_RSK_WORD_LENGTH)
    added_row: StrictInt = Field(ge=0, le=MAX_RSK_WORD_LENGTH - 1)
    added_column: StrictInt = Field(ge=0, le=MAX_RSK_WORD_LENGTH - 1)
    added_entry: StrictInt = Field(ge=1, le=MAX_ALPHABET_SIZE)
    row_lengths: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_RSK_WORD_LENGTH
    )

    @model_validator(mode="after")
    def require_contiguous_row_path(self) -> Self:
        if tuple(step.row for step in self.bump_path) != tuple(
            range(len(self.bump_path))
        ) or self.added_row != len(self.bump_path):
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_trace_row_path",
                "bump steps must visit consecutive rows from zero and end below them",
            )
        if len(self.bump_path) > self.position - 1:
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_trace_path_too_long",
                "an insertion cannot bump more cells than the preceding word length",
            )
        if self.added_row >= self.position or self.added_column >= self.position:
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_trace_cell_out_of_range",
                "an insertion cell must lie within its prefix's cell bound",
            )
        if (
            sum(self.row_lengths) != self.position
            or any(length <= 0 for length in self.row_lengths)
            or any(
                self.row_lengths[index] < self.row_lengths[index + 1]
                for index in range(len(self.row_lengths) - 1)
            )
            or self.added_row >= len(self.row_lengths)
            or self.row_lengths[self.added_row] != self.added_column + 1
        ):
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_trace_shape",
                "each event must retain partition row lengths after insertion",
            )
        if any(
            step.column >= self.position
            or step.row >= len(self.row_lengths)
            or self.row_lengths[step.row] <= step.column
            for step in self.bump_path
        ):
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_trace_cell_out_of_range",
                "a bump cell must lie within its prefix's cell bound",
            )
        return self


class RSKReverseBumpStep(StrictModel):
    """One reverse row replacement, with zero-based cell coordinates."""

    row: StrictInt = Field(ge=0, le=MAX_RSK_WORD_LENGTH - 1)
    column: StrictInt = Field(ge=0, le=MAX_RSK_WORD_LENGTH - 1)
    displaced_entry: StrictInt = Field(ge=1, le=MAX_ALPHABET_SIZE)


class RSKReverseInsertionEvent(StrictModel):
    """Removal of one recording label and its reverse bump path."""

    position: StrictInt = Field(ge=1, le=MAX_RSK_WORD_LENGTH)
    letter: Symbol
    removed_row: StrictInt = Field(ge=0, le=MAX_RSK_WORD_LENGTH - 1)
    removed_column: StrictInt = Field(ge=0, le=MAX_RSK_WORD_LENGTH - 1)
    removed_entry: StrictInt = Field(ge=1, le=MAX_ALPHABET_SIZE)
    output_entry: StrictInt = Field(ge=1, le=MAX_ALPHABET_SIZE)
    reverse_bump_path: tuple[RSKReverseBumpStep, ...] = Field(
        max_length=MAX_RSK_WORD_LENGTH
    )
    row_lengths: tuple[StrictInt, ...] = Field(max_length=MAX_RSK_WORD_LENGTH)

    @model_validator(mode="after")
    def require_reverse_path_and_shape(self) -> Self:
        if tuple(step.row for step in self.reverse_bump_path) != tuple(
            range(self.removed_row - 1, -1, -1)
        ):
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_reverse_trace_row_path",
                "reverse bump steps must visit every preceding row in descending order",
            )
        if self.output_entry != (
            self.reverse_bump_path[-1].displaced_entry
            if self.reverse_bump_path
            else self.removed_entry
        ):
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_reverse_trace_output_entry",
                "the output entry must be the final carried entry",
            )
        if (
            sum(self.row_lengths) != self.position - 1
            or any(length <= 0 for length in self.row_lengths)
            or any(
                left < right
                for left, right in zip(
                    self.row_lengths, self.row_lengths[1:], strict=False
                )
            )
        ):
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_reverse_trace_shape",
                "row lengths must describe the partition after removal",
            )
        if self.removed_row >= len(self.row_lengths) + 1:
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_reverse_trace_cell",
                "the removed cell row is outside the prior shape",
            )
        expected_removed_column = (
            self.row_lengths[self.removed_row]
            if self.removed_row < len(self.row_lengths)
            else 0
        )
        if self.removed_column != expected_removed_column:
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_reverse_trace_cell",
                "the removed column must be the corner immediately beyond the resulting row",
            )
        if any(
            step.row >= len(self.row_lengths)
            or step.column >= self.row_lengths[step.row]
            for step in self.reverse_bump_path
        ):
            raise PydanticCustomError(
                "algebraic_combinatorics.rsk_reverse_trace_cell",
                "a reverse bump cell must lie within the resulting shape",
            )
        return self


__all__ = [
    "MAX_RSK_ALPHABET_RANK_DIGITS",
    "MAX_RSK_ROW_SEARCH_COMPARISONS",
    "MAX_RSK_WORD_LENGTH",
    "MAX_RSK_WORD_PAYLOAD_SCALARS",
    "RSKBumpStep",
    "RSKConvention",
    "RSKInsertionEvent",
    "RSKReverseBumpStep",
    "RSKReverseInsertionEvent",
    "RSKTableauPair",
]
