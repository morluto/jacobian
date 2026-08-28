"""Canonical values for bounded RSK correspondences."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
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
# can use MAX_SYMBOL_LENGTH Unicode scalar values per symbol, each encoded in
# four UTF-8 bytes.
MAX_RSK_WORD_BYTES = (MAX_RSK_WORD_LENGTH + MAX_ALPHABET_SIZE) * MAX_SYMBOL_LENGTH * 4
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
        if any(
            entry > len(self.alphabet)
            for row in self.insertion_tableau.rows
            for entry in row
        ):
            raise PydanticCustomError(
                "algebraic_combinatorics.insertion_entry_out_of_range",
                "insertion tableau entry is outside the ordered alphabet",
            )
        return self


__all__ = [
    "MAX_RSK_ROW_SEARCH_COMPARISONS",
    "MAX_RSK_WORD_BYTES",
    "MAX_RSK_WORD_LENGTH",
    "RSKConvention",
    "RSKTableauPair",
]
