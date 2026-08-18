"""Typed wire contracts for exact combinatorics-on-words operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.words.operations import factors_of_length, incidence_matrix, periods
from jacobian.math.words.values import FiniteWord, WordMorphism


class FactorsLengthRequest(StrictModel):
    """Enumerate all distinct factors of one valid length."""

    word: FiniteWord
    factor_length: int = Field(ge=0)

    @model_validator(mode="after")
    def require_bounded_factor_length(self) -> Self:
        if self.factor_length > len(self.word.letters):
            raise ValueError("factor_length must not exceed the word length")
        return self


class FactorsLengthResult(FactorsLengthRequest):
    """Complete factor enumeration, ordered by first occurrence."""

    factors: tuple[tuple[str, ...], ...]
    occurrences: tuple[tuple[int, ...], ...]
    multiplicities: tuple[int, ...]
    first_occurrence: tuple[int, ...]
    distinct_count: int = Field(ge=0)
    complete: Literal[True] = True
    scope: Literal["ALL_CONTIGUOUS_FACTORS_OF_REQUESTED_LENGTH"] = (
        "ALL_CONTIGUOUS_FACTORS_OF_REQUESTED_LENGTH"
    )
    method: Literal["EXACT_SLIDING_WINDOW_ENUMERATION"] = (
        "EXACT_SLIDING_WINDOW_ENUMERATION"
    )

    @model_validator(mode="after")
    def bind_exact_factor_enumeration(self) -> Self:
        expected = factors_of_length(self.word, self.factor_length)
        expected_occurrences = expected.occurrences
        if (
            self.factors != expected.factors
            or self.occurrences != expected_occurrences
            or self.multiplicities
            != tuple(len(indices) for indices in expected_occurrences)
            or self.first_occurrence
            != tuple(indices[0] for indices in expected_occurrences)
            or self.distinct_count != len(expected.factors)
        ):
            raise ValueError("factor result is not bound to the requested word")
        return self


class PeriodsRequest(StrictModel):
    """Compute all overlap periods of a finite word."""

    word: FiniteWord


class PeriodsResult(PeriodsRequest):
    """Complete overlap-period profile and proper-power primitivity."""

    periods: tuple[int, ...]
    least_period: int = Field(ge=0)
    is_primitive: bool
    complete: Literal[True] = True
    method: Literal["EXACT_OVERLAP_COMPARISON"] = "EXACT_OVERLAP_COMPARISON"
    primitive_convention: Literal["NOT_A_NONTRIVIAL_INTEGER_POWER"] = (
        "NOT_A_NONTRIVIAL_INTEGER_POWER"
    )
    empty_word_convention: Literal["NO_POSITIVE_PERIOD_AND_NOT_PRIMITIVE"] = (
        "NO_POSITIVE_PERIOD_AND_NOT_PRIMITIVE"
    )

    @model_validator(mode="after")
    def bind_exact_period_profile(self) -> Self:
        expected = periods(self.word)
        if (
            self.periods != expected.periods
            or self.least_period != expected.least_period
            or self.is_primitive != expected.primitive
        ):
            raise ValueError("period result is not bound to the requested word")
        return self


class IncidenceMatrixRequest(StrictModel):
    """Compute the incidence matrix of a finite word morphism."""

    morphism: WordMorphism


class IncidenceMatrixResult(IncidenceMatrixRequest):
    """Exact target-by-source incidence matrix."""

    matrix: tuple[tuple[int, ...], ...]
    complete: Literal[True] = True
    method: Literal["EXACT_SYMBOL_COUNTING"] = "EXACT_SYMBOL_COUNTING"
    orientation: Literal["ROWS_TARGET_COLUMNS_SOURCE"] = "ROWS_TARGET_COLUMNS_SOURCE"

    @model_validator(mode="after")
    def bind_exact_incidence_matrix(self) -> Self:
        if self.matrix != incidence_matrix(self.morphism):
            raise ValueError("incidence matrix is not bound to the requested morphism")
        return self


__all__ = [
    "FactorsLengthRequest",
    "FactorsLengthResult",
    "IncidenceMatrixRequest",
    "IncidenceMatrixResult",
    "PeriodsRequest",
    "PeriodsResult",
]
