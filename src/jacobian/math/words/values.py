"""Provider-independent values for bounded combinatorics on words."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_WORD_LENGTH = 500
MAX_ALPHABET_SIZE = 50
MAX_SYMBOL_LENGTH = 64
MAX_MORPHISM_IMAGE_LENGTH = 10_000
MAX_MORPHISM_OUTPUT_LENGTH = MAX_WORD_LENGTH

Symbol = Annotated[str, Field(min_length=1, max_length=MAX_SYMBOL_LENGTH)]


class FiniteWord(StrictModel):
    alphabet: tuple[Symbol, ...] = Field(min_length=1, max_length=MAX_ALPHABET_SIZE)
    letters: tuple[str, ...] = Field(max_length=MAX_WORD_LENGTH)

    @model_validator(mode="after")
    def require_word_over_ordered_alphabet(self) -> Self:
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet symbols must be distinct")
        if any(letter not in self.alphabet for letter in self.letters):
            raise ValueError("word letter is outside the declared alphabet")
        return self


class WordMorphism(StrictModel):
    source_alphabet: tuple[Symbol, ...] = Field(
        min_length=1, max_length=MAX_ALPHABET_SIZE
    )
    target_alphabet: tuple[Symbol, ...] = Field(
        min_length=1, max_length=MAX_ALPHABET_SIZE
    )
    images: tuple[tuple[str, ...], ...] = Field(
        min_length=1, max_length=MAX_ALPHABET_SIZE
    )

    @model_validator(mode="after")
    def require_total_bounded_morphism(self) -> Self:
        if len(set(self.source_alphabet)) != len(self.source_alphabet):
            raise ValueError("source alphabet symbols must be distinct")
        if len(set(self.target_alphabet)) != len(self.target_alphabet):
            raise ValueError("target alphabet symbols must be distinct")
        if len(self.images) != len(self.source_alphabet):
            raise ValueError("morphism must have one image per source symbol")
        if any(len(image) > MAX_MORPHISM_IMAGE_LENGTH for image in self.images):
            raise ValueError("morphism image exceeds the length bound")
        if any(
            letter not in self.target_alphabet
            for image in self.images
            for letter in image
        ):
            raise ValueError("morphism image uses a symbol outside the target alphabet")
        return self


__all__ = [
    "MAX_ALPHABET_SIZE",
    "MAX_MORPHISM_IMAGE_LENGTH",
    "MAX_MORPHISM_OUTPUT_LENGTH",
    "MAX_SYMBOL_LENGTH",
    "MAX_WORD_LENGTH",
    "FiniteWord",
    "Symbol",
    "WordMorphism",
]
