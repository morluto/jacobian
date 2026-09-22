from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.combinatorics.algebraic.values import RSKConvention
from jacobian.math.combinatorics.symmetric_functions.values import (
    IntegerPartition,
    SemistandardYoungTableau,
)
from jacobian.math.logic.languages.words.values import FiniteWord

MAX_BIWORD_MASS = 500
MAX_BIWORD_AXIS = 50
MAX_BIWORD_LABEL_BYTES = 2_048
MAX_BIWORD_CELLS = 2_500


def _e(r: str, m: str) -> PydanticCustomError:
    return PydanticCustomError(f"algebraic_combinatorics.{r}", m)


class Biword(StrictModel):
    top_alphabet: tuple[str, ...] = Field(max_length=MAX_BIWORD_AXIS)
    bottom_alphabet: tuple[str, ...] = Field(max_length=MAX_BIWORD_AXIS)
    top: tuple[str, ...] = Field(max_length=MAX_BIWORD_MASS)
    bottom: tuple[str, ...] = Field(max_length=MAX_BIWORD_MASS)

    @model_validator(mode="after")
    def canonical(self) -> Self:
        if len(set(self.top_alphabet)) != len(self.top_alphabet) or len(
            set(self.bottom_alphabet)
        ) != len(self.bottom_alphabet):
            raise _e("biword_alphabet", "biword alphabets must be unique")
        try:
            label_bytes = sum(
                len(x.encode("utf-8"))
                for x in (*self.top_alphabet, *self.bottom_alphabet)
            )
        except UnicodeEncodeError:
            raise _e("biword_alphabet", "biword labels must be UTF-8") from None
        if label_bytes > MAX_BIWORD_LABEL_BYTES:
            raise _e("biword_alphabet", "biword labels exceed the byte envelope")
        if len(self.top) != len(self.bottom) or len(self.top) > MAX_BIWORD_MASS:
            raise _e("biword_length", "biword rows must have equal bounded length")
        tr = {x: i for i, x in enumerate(self.top_alphabet)}
        br = {x: i for i, x in enumerate(self.bottom_alphabet)}
        if any(x not in tr for x in self.top) or any(x not in br for x in self.bottom):
            raise _e("biword_symbol", "biword symbols must belong to their alphabets")
        if tuple(
            (tr[a], br[b]) for a, b in zip(self.top, self.bottom, strict=True)
        ) != tuple(
            sorted((tr[a], br[b]) for a, b in zip(self.top, self.bottom, strict=True))
        ):
            raise _e(
                "biword_order",
                "biword must be lexicographically sorted by (top,bottom)",
            )
        return self


class NonnegativeIntegerMatrix(StrictModel):
    row_labels: tuple[str, ...] = Field(max_length=MAX_BIWORD_AXIS)
    column_labels: tuple[str, ...] = Field(max_length=MAX_BIWORD_AXIS)
    entries: tuple[tuple[ExactInteger, ...], ...]

    @model_validator(mode="after")
    def shape(self) -> Self:
        if len(set(self.row_labels)) != len(self.row_labels) or len(
            set(self.column_labels)
        ) != len(self.column_labels):
            raise _e("matrix_labels", "matrix labels must be unique")
        try:
            label_bytes = sum(
                len(x.encode("utf-8")) for x in (*self.row_labels, *self.column_labels)
            )
        except UnicodeEncodeError:
            raise _e("matrix_labels", "matrix labels must be UTF-8") from None
        if label_bytes > MAX_BIWORD_LABEL_BYTES:
            raise _e("matrix_labels", "matrix labels exceed the byte envelope")
        if len(self.row_labels) * len(self.column_labels) > MAX_BIWORD_CELLS:
            raise _e(
                "matrix_cells", "matrix cell axes exceed the materialization envelope"
            )
        if len(self.entries) != len(self.row_labels) or any(
            len(r) != len(self.column_labels) for r in self.entries
        ):
            raise _e("matrix_shape", "matrix entries must match labelled axes")
        if (
            any(x < 0 for r in self.entries for x in r)
            or sum(sum(r) for r in self.entries) > MAX_BIWORD_MASS
        ):
            raise _e(
                "matrix_nonnegative",
                "matrix cells must be nonnegative and total mass bounded",
            )
        return self


class BiwordRSKPair(StrictModel):
    top_alphabet: tuple[str, ...] = Field(max_length=MAX_BIWORD_AXIS)
    bottom_alphabet: tuple[str, ...] = Field(max_length=MAX_BIWORD_AXIS)
    insertion_tableau: SemistandardYoungTableau
    recording_tableau: SemistandardYoungTableau
    shape: IntegerPartition
    source_kind: Literal["BIWORD", "MATRIX"]
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"

    @model_validator(mode="after")
    def compatible(self) -> Self:
        if (
            self.insertion_tableau.shape != self.shape
            or self.recording_tableau.shape != self.shape
        ):
            raise _e("biword_shape", "tableaux and shape must agree")
        if len(set(self.top_alphabet)) != len(self.top_alphabet) or len(
            set(self.bottom_alphabet)
        ) != len(self.bottom_alphabet):
            raise _e("biword_alphabet", "pair alphabets must be unique")
        if any(
            entry < 1 or entry > len(self.bottom_alphabet)
            for row in self.insertion_tableau.rows
            for entry in row
        ):
            raise _e(
                "biword_content", "insertion entries must address the bottom alphabet"
            )
        if any(
            entry < 1 or entry > len(self.top_alphabet)
            for row in self.recording_tableau.rows
            for entry in row
        ):
            raise _e(
                "biword_content", "recording entries must address the top alphabet"
            )
        return self


class BiwordNormalizeRequest(StrictModel):
    """An unsorted biword admitted before the normalization sort."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Normalize at most 500 paired rows over axes of at most 50 labels "
                "and 2,048 UTF-8 label bytes; the exact stable permutation has one "
                "entry per input row."
            )
        }
    )

    top_alphabet: tuple[str, ...] = Field(max_length=MAX_BIWORD_AXIS)
    bottom_alphabet: tuple[str, ...] = Field(max_length=MAX_BIWORD_AXIS)
    top: tuple[str, ...] = Field(max_length=MAX_BIWORD_MASS)
    bottom: tuple[str, ...] = Field(max_length=MAX_BIWORD_MASS)

    @model_validator(mode="after")
    def bounded_source(self) -> Self:
        if len(set(self.top_alphabet)) != len(self.top_alphabet) or len(
            set(self.bottom_alphabet)
        ) != len(self.bottom_alphabet):
            raise _e("biword_alphabet", "biword alphabets must be unique")
        try:
            label_bytes = sum(
                len(x.encode("utf-8"))
                for x in (*self.top_alphabet, *self.bottom_alphabet)
            )
        except UnicodeEncodeError:
            raise _e("biword_alphabet", "biword labels must be UTF-8") from None
        if label_bytes > MAX_BIWORD_LABEL_BYTES:
            raise _e("biword_alphabet", "biword labels exceed the byte envelope")
        if len(self.top) != len(self.bottom):
            raise _e("biword_length", "biword rows must have equal bounded length")
        tr = set(self.top_alphabet)
        br = set(self.bottom_alphabet)
        if any(x not in tr for x in self.top) or any(x not in br for x in self.bottom):
            raise _e("biword_symbol", "biword symbols must belong to their alphabets")
        return self


class BiwordNormalizeResult(StrictModel):
    source_top: tuple[str, ...]
    source_bottom: tuple[str, ...]
    biword: Biword
    permutation: tuple[int, ...]


class BiwordRSKRequest(StrictModel):
    biword: Biword
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"


class MatrixRSKRequest(StrictModel):
    matrix: NonnegativeIntegerMatrix
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"


class InverseBiwordRSKRequest(StrictModel):
    pair: BiwordRSKPair
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"


class InverseMatrixRSKRequest(StrictModel):
    pair: BiwordRSKPair
    row_labels: tuple[str, ...]
    column_labels: tuple[str, ...]
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"


class GreeneRequest(StrictModel):
    word: FiniteWord
    k: int = Field(ge=1, le=20)


class GreeneResult(StrictModel):
    word: FiniteWord
    shape: IntegerPartition
    increasing_totals: tuple[int, ...]
    decreasing_totals: tuple[int, ...]
    k: int


__all__ = [
    "Biword",
    "BiwordNormalizeRequest",
    "BiwordNormalizeResult",
    "BiwordRSKPair",
    "BiwordRSKRequest",
    "GreeneRequest",
    "GreeneResult",
    "InverseBiwordRSKRequest",
    "InverseMatrixRSKRequest",
    "MatrixRSKRequest",
    "NonnegativeIntegerMatrix",
]
