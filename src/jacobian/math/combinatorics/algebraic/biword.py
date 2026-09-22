from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
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
MAX_BIWORD_ALPHABET = 64
MAX_BIWORD_LABEL_BYTES = 4096


def _e(r: str, m: str) -> PydanticCustomError:
    return PydanticCustomError(f"algebraic_combinatorics.{r}", m)


class Biword(StrictModel):
    top_alphabet: tuple[str, ...] = Field(max_length=MAX_BIWORD_ALPHABET)
    bottom_alphabet: tuple[str, ...] = Field(max_length=MAX_BIWORD_ALPHABET)
    top: tuple[str, ...]
    bottom: tuple[str, ...]

    @model_validator(mode="after")
    def canonical(self) -> Self:
        if (
            sum(
                len(label.encode("utf-8"))
                for label in (*self.top_alphabet, *self.bottom_alphabet)
            )
            > MAX_BIWORD_LABEL_BYTES
        ):
            raise _e(
                "biword_alphabet",
                f"biword alphabet labels are limited to {MAX_BIWORD_LABEL_BYTES} UTF-8 bytes",
            )
        if len(set(self.top_alphabet)) != len(self.top_alphabet) or len(
            set(self.bottom_alphabet)
        ) != len(self.bottom_alphabet):
            raise _e("biword_alphabet", "biword alphabets must be unique")
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
    row_labels: tuple[str, ...] = Field(max_length=MAX_BIWORD_ALPHABET)
    column_labels: tuple[str, ...] = Field(max_length=MAX_BIWORD_ALPHABET)
    entries: tuple[tuple[ExactInteger, ...], ...]

    @model_validator(mode="after")
    def shape(self) -> Self:
        if (
            sum(
                len(label.encode("utf-8"))
                for label in (*self.row_labels, *self.column_labels)
            )
            > MAX_BIWORD_LABEL_BYTES
        ):
            raise _e(
                "matrix_labels",
                f"matrix labels are limited to {MAX_BIWORD_LABEL_BYTES} UTF-8 bytes",
            )
        if len(set(self.row_labels)) != len(self.row_labels) or len(
            set(self.column_labels)
        ) != len(self.column_labels):
            raise _e("matrix_labels", "matrix labels must be unique")
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
    top_alphabet: tuple[str, ...] = Field(max_length=MAX_BIWORD_ALPHABET)
    bottom_alphabet: tuple[str, ...] = Field(max_length=MAX_BIWORD_ALPHABET)
    insertion_tableau: SemistandardYoungTableau
    recording_tableau: SemistandardYoungTableau
    shape: IntegerPartition
    source_kind: Literal["BIWORD", "MATRIX"]
    convention: RSKConvention = "ROW_INSERTION_RSK_V1"

    @model_validator(mode="after")
    def compatible(self) -> Self:
        if (
            sum(
                len(label.encode("utf-8"))
                for label in (*self.top_alphabet, *self.bottom_alphabet)
            )
            > MAX_BIWORD_LABEL_BYTES
        ):
            raise _e(
                "biword_alphabet",
                f"pair alphabet labels are limited to {MAX_BIWORD_LABEL_BYTES} UTF-8 bytes",
            )
        if (
            self.insertion_tableau.shape != self.shape
            or self.recording_tableau.shape != self.shape
        ):
            raise _e("biword_shape", "tableaux and shape must agree")
        if len(set(self.top_alphabet)) != len(self.top_alphabet) or len(
            set(self.bottom_alphabet)
        ) != len(self.bottom_alphabet):
            raise _e("biword_alphabet", "pair alphabets must be unique")
        top_limit = len(self.top_alphabet)
        bottom_limit = len(self.bottom_alphabet)
        insertion = tuple(x for row in self.insertion_tableau.rows for x in row)
        recording = tuple(x for row in self.recording_tableau.rows for x in row)
        if any(type(x) is not int or x < 1 or x > bottom_limit for x in insertion):
            raise _e(
                "biword_insertion_entries",
                "insertion entries must be one-based bottom-alphabet ranks",
            )
        if any(type(x) is not int or x < 1 or x > top_limit for x in recording):
            raise _e(
                "biword_recording_entries",
                "recording entries must be one-based top-alphabet ranks",
            )
        if (insertion or recording) and (not top_limit or not bottom_limit):
            raise _e("biword_empty_alphabet", "nonempty tableaux need both alphabets")
        if len(insertion) != len(recording):
            raise _e("biword_cells", "tableaux must have the same cell count")
        return self


class BiwordNormalizeRequest(StrictModel):
    top_alphabet: tuple[str, ...] = Field(max_length=MAX_BIWORD_ALPHABET)
    bottom_alphabet: tuple[str, ...] = Field(max_length=MAX_BIWORD_ALPHABET)
    top: tuple[str, ...] = Field(max_length=MAX_BIWORD_MASS)
    bottom: tuple[str, ...] = Field(max_length=MAX_BIWORD_MASS)

    @model_validator(mode="after")
    def bounded(self) -> Self:
        if (
            sum(
                len(label.encode("utf-8"))
                for label in (*self.top_alphabet, *self.bottom_alphabet)
            )
            > MAX_BIWORD_LABEL_BYTES
        ):
            raise _e(
                "biword_alphabet",
                f"biword alphabet labels are limited to {MAX_BIWORD_LABEL_BYTES} UTF-8 bytes",
            )
        if len(self.top) != len(self.bottom):
            raise _e("biword_length", "biword rows must have equal length")
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
    row_labels: tuple[str, ...] = Field(max_length=MAX_BIWORD_ALPHABET)
    column_labels: tuple[str, ...] = Field(max_length=MAX_BIWORD_ALPHABET)
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
