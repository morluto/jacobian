"""Canonical values for bounded RSK correspondences."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
    StandardYoungTableau,
)
from jacobian.math.words.values import MAX_ALPHABET_SIZE, Symbol

MAX_RSK_WORD_LENGTH = 50
# A word at the length boundary and all 50 declared symbols can each use the
# FiniteWord maximum of 64 Unicode scalar values, each encoded in four bytes.
MAX_RSK_WORD_BYTES = 25_600
RSKConvention = Literal["ROW_INSERTION_RSK_V1"]


class RSKTableauPair(StrictModel):
    """The compact ordinary-word RSK image under row insertion.

    Insertion-tableau entries are one-based ranks in ``alphabet``.  The
    alphabet therefore remains attached to the pair and makes inverse RSK an
    exact operation even when symbols are not integers.  The common shape has
    at most 50 cells.
    """

    alphabet: tuple[Symbol, ...] = Field(
        min_length=1,
        max_length=MAX_ALPHABET_SIZE,
        description=(
            "The exact ordered source alphabet; insertion-tableau entry i "
            "denotes alphabet[i - 1]."
        ),
        json_schema_extra={"uniqueItems": True},
    )
    insertion_tableau: SemistandardYoungTableau
    recording_tableau: StandardYoungTableau
    shape: IntegerPartition = Field(
        description=(
            "The common tableau shape, required to equal both derived row-length "
            f"partitions and to contain at most {MAX_RSK_WORD_LENGTH} cells."
        )
    )
    source_kind: Literal["WORD"] = "WORD"
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"

    @model_validator(mode="after")
    def require_compatible_word_pair(self) -> Self:
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet symbols must be distinct")
        if self.insertion_tableau.shape != self.shape:
            raise ValueError("insertion tableau shape must equal the common shape")
        if self.recording_tableau.shape != self.shape:
            raise ValueError("recording tableau shape must equal the common shape")
        if sum(self.shape.parts) > MAX_RSK_WORD_LENGTH:
            raise ValueError(
                f"RSK tableau pair size must not exceed {MAX_RSK_WORD_LENGTH}"
            )
        if any(
            entry > len(self.alphabet)
            for row in self.insertion_tableau.rows
            for entry in row
        ):
            raise ValueError("insertion tableau entry is outside the ordered alphabet")
        return self


__all__ = [
    "MAX_RSK_WORD_BYTES",
    "MAX_RSK_WORD_LENGTH",
    "RSKConvention",
    "RSKTableauPair",
]
