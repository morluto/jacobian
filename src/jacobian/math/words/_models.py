"""Typed wire contracts for combinatorics on words operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_WORD_LEN = 500
MAX_ALPHABET = 50
MAX_MORPHISM_LEN = 100
MAX_ITERATE_LEN = 10_000
MAX_FACTORS = 5_000


# -- Word models -------------------------------------------------------------


class FiniteWordRequest(StrictModel):
    """A finite word over a finite ordered alphabet."""

    alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    word: tuple[str, ...] = Field(max_length=MAX_WORD_LEN)

    @model_validator(mode="after")
    def validate_letters(self) -> Self:
        valid = set(self.alphabet)
        for letter in self.word:
            if letter not in valid:
                raise ValueError(f"letter {letter!r} is not in the alphabet")
        return self


class FactorsLengthRequest(StrictModel):
    """Find all distinct factors of a given length."""

    alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    word: tuple[str, ...] = Field(max_length=MAX_WORD_LEN)
    factor_length: int = Field(ge=0, le=MAX_WORD_LEN)

    @model_validator(mode="after")
    def validate_letters(self) -> Self:
        valid = set(self.alphabet)
        for letter in self.word:
            if letter not in valid:
                raise ValueError(f"letter {letter!r} is not in the alphabet")
        return self


class FactorOccurrencesRequest(StrictModel):
    """Find all occurrences of a pattern in a word."""

    alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    word: tuple[str, ...] = Field(max_length=MAX_WORD_LEN)
    pattern: tuple[str, ...] = Field(max_length=MAX_WORD_LEN)

    @model_validator(mode="after")
    def validate_letters(self) -> Self:
        valid = set(self.alphabet)
        for letter in self.word:
            if letter not in valid:
                raise ValueError(f"letter {letter!r} is not in the alphabet")
        for letter in self.pattern:
            if letter not in valid:
                raise ValueError(f"pattern letter {letter!r} is not in the alphabet")
        return self


class PeriodsRequest(StrictModel):
    """Compute all periods of a finite word."""

    alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    word: tuple[str, ...] = Field(max_length=MAX_WORD_LEN)

    @model_validator(mode="after")
    def validate_letters(self) -> Self:
        valid = set(self.alphabet)
        for letter in self.word:
            if letter not in valid:
                raise ValueError(f"letter {letter!r} is not in the alphabet")
        return self


class PrimitiveRootRequest(StrictModel):
    """Compute the primitive root of a finite word."""

    alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    word: tuple[str, ...] = Field(max_length=MAX_WORD_LEN)

    @model_validator(mode="after")
    def validate_letters(self) -> Self:
        valid = set(self.alphabet)
        for letter in self.word:
            if letter not in valid:
                raise ValueError(f"letter {letter!r} is not in the alphabet")
        return self


class ConjugatesRequest(StrictModel):
    """Compute all cyclic conjugates of a finite word."""

    alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    word: tuple[str, ...] = Field(max_length=MAX_WORD_LEN)

    @model_validator(mode="after")
    def validate_letters(self) -> Self:
        valid = set(self.alphabet)
        for letter in self.word:
            if letter not in valid:
                raise ValueError(f"letter {letter!r} is not in the alphabet")
        return self


class ParikhRequest(StrictModel):
    """Compute the Parikh vector of a finite word."""

    alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    word: tuple[str, ...] = Field(max_length=MAX_WORD_LEN)

    @model_validator(mode="after")
    def validate_letters(self) -> Self:
        valid = set(self.alphabet)
        for letter in self.word:
            if letter not in valid:
                raise ValueError(f"letter {letter!r} is not in the alphabet")
        return self


class PrefixFunctionRequest(StrictModel):
    """Compute the Knuth-Morris-Pratt prefix function of a finite word."""

    alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    word: tuple[str, ...] = Field(max_length=MAX_WORD_LEN)

    @model_validator(mode="after")
    def validate_letters(self) -> Self:
        valid = set(self.alphabet)
        for letter in self.word:
            if letter not in valid:
                raise ValueError(f"letter {letter!r} is not in the alphabet")
        return self


# -- Morphism models ---------------------------------------------------------


class MorphismApplyRequest(StrictModel):
    """Apply a word morphism to a finite word."""

    source_alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    target_alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    images: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    word: tuple[str, ...] = Field(max_length=MAX_WORD_LEN)

    @model_validator(mode="after")
    def validate_morphism(self) -> Self:
        if len(self.images) != len(self.source_alphabet):
            raise ValueError("images must have one entry per source letter")
        for img in self.images:
            if len(img) > MAX_MORPHISM_LEN:
                raise ValueError(
                    f"image length must be at most {MAX_MORPHISM_LEN}"
                )
            for letter in img:
                if letter not in set(self.target_alphabet):
                    raise ValueError(
                        f"image letter {letter!r} is not in the target alphabet"
                    )
        for letter in self.word:
            if letter not in set(self.source_alphabet):
                raise ValueError(f"word letter {letter!r} is not in the source alphabet")
        return self


class MorphismComposeRequest(StrictModel):
    """Compose two word morphisms: tau(sigma(w))."""

    source_alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    middle_alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    target_alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    sigma_images: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    tau_images: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=MAX_ALPHABET)

    @model_validator(mode="after")
    def validate_compose(self) -> Self:
        if len(self.sigma_images) != len(self.source_alphabet):
            raise ValueError("sigma_images must have one entry per source letter")
        if len(self.tau_images) != len(self.middle_alphabet):
            raise ValueError("tau_images must have one entry per middle letter")
        middle_set = set(self.middle_alphabet)
        for img in self.sigma_images:
            for letter in img:
                if letter not in middle_set:
                    raise ValueError(
                        f"sigma image letter {letter!r} is not in the middle alphabet"
                    )
        target_set = set(self.target_alphabet)
        for img in self.tau_images:
            for letter in img:
                if letter not in target_set:
                    raise ValueError(
                        f"tau image letter {letter!r} is not in the target alphabet"
                    )
        return self


class IncidenceMatrixRequest(StrictModel):
    """Compute the incidence matrix of a word morphism."""

    source_alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    target_alphabet: tuple[str, ...] = Field(min_length=1, max_length=MAX_ALPHABET)
    images: tuple[tuple[str, ...], ...] = Field(min_length=1, max_length=MAX_ALPHABET)

    @model_validator(mode="after")
    def validate_morphism(self) -> Self:
        if len(self.images) != len(self.source_alphabet):
            raise ValueError("images must have one entry per source letter")
        target_set = set(self.target_alphabet)
        for img in self.images:
            for letter in img:
                if letter not in target_set:
                    raise ValueError(
                        f"image letter {letter!r} is not in the target alphabet"
                    )
        return self


# -- Results -----------------------------------------------------------------


class FactorsLengthResult(StrictModel):
    """Distinct factors of a given length."""

    factor_length: int = Field(ge=0)
    factors: tuple[tuple[str, ...], ...]
    occurrences: tuple[tuple[int, ...], ...]
    multiplicities: tuple[int, ...]
    first_occurrence: tuple[int, ...]
    distinct_count: int = Field(ge=0)


class FactorOccurrencesResult(StrictModel):
    """Occurrences of a pattern in a word."""

    pattern: tuple[str, ...]
    occurrences: tuple[int, ...]
    count: int = Field(ge=0)


class PeriodsResult(StrictModel):
    """All periods of a finite word."""

    periods: tuple[int, ...]
    least_period: int = Field(ge=0)
    is_primitive: bool


class PrimitiveRootResult(StrictModel):
    """Primitive root of a finite word."""

    root: tuple[str, ...]
    exponent: int = Field(ge=1)


class ConjugatesResult(StrictModel):
    """All cyclic conjugates of a finite word."""

    conjugates: tuple[tuple[str, ...], ...]
    least_lexicographic: tuple[str, ...]
    rotation_index: tuple[int, ...]


class ParikhResult(StrictModel):
    """Parikh vector of a finite word."""

    parikh_vector: tuple[int, ...]
    length: int = Field(ge=0)
    support: tuple[str, ...]


class PrefixFunctionResult(StrictModel):
    """KMP prefix function (border table)."""

    prefix_function: tuple[int, ...]
    border_lengths: tuple[int, ...]


class MorphismApplyResult(StrictModel):
    """Result of applying a morphism to a word."""

    image: tuple[str, ...]
    length: int = Field(ge=0)


class MorphismComposeResult(StrictModel):
    """Result of composing two morphisms."""

    images: tuple[tuple[str, ...], ...]


class IncidenceMatrixResult(StrictModel):
    """Incidence matrix of a morphism."""

    matrix: tuple[tuple[int, ...], ...]
    source_alphabet: tuple[str, ...]
    target_alphabet: tuple[str, ...]


__all__ = [
    "ConjugatesRequest",
    "ConjugatesResult",
    "FactorOccurrencesRequest",
    "FactorOccurrencesResult",
    "FactorsLengthRequest",
    "FactorsLengthResult",
    "FiniteWordRequest",
    "IncidenceMatrixRequest",
    "IncidenceMatrixResult",
    "MorphismApplyRequest",
    "MorphismApplyResult",
    "MorphismComposeRequest",
    "MorphismComposeResult",
    "ParikhRequest",
    "ParikhResult",
    "PeriodsRequest",
    "PeriodsResult",
    "PrefixFunctionRequest",
    "PrefixFunctionResult",
    "PrimitiveRootRequest",
    "PrimitiveRootResult",
]
