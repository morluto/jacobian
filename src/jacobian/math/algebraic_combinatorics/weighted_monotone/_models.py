"""Typed contracts for the weighted monotone subsequence endpoint profile."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.logic.languages.words.values import FiniteWord

MAX_WORD_LENGTH = 100


class WeightedOrderedWord(StrictModel):
    """A finite word with a nonnegative rational weight per position."""

    word: FiniteWord
    weights: tuple[CanonicalRational, ...] = Field(max_length=MAX_WORD_LENGTH)

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        if len(self.weights) != len(self.word.letters):
            raise PydanticCustomError(
                "weighted_word.length_mismatch",
                "weights length must match word length",
            )
        for w in self.weights:
            if w.as_fraction() < 0:
                raise PydanticCustomError(
                    "weighted_word.negative_weight",
                    "all weights must be nonnegative",
                )
        return self


class EndpointProfileRequest(StrictModel):
    """Request for the weighted monotone subsequence endpoint profiles."""

    source: WeightedOrderedWord


class EndpointProfileEntry(StrictModel):
    """One position's endpoint values."""

    position: int
    letter: str
    weight: CanonicalRational
    increasing_value: CanonicalRational
    decreasing_value: CanonicalRational


class EndpointProfileResult(StrictModel):
    """The complete endpoint profiles S_i and T_i."""

    source: WeightedOrderedWord
    entries: tuple[EndpointProfileEntry, ...]


__all__ = [
    "MAX_WORD_LENGTH",
    "EndpointProfileEntry",
    "EndpointProfileRequest",
    "EndpointProfileResult",
    "WeightedOrderedWord",
]
