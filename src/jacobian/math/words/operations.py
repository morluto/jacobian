"""Exact bounded native kernels for combinatorics on words."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.math.words.values import (
    MAX_MORPHISM_OUTPUT_LENGTH,
    FiniteWord,
    WordMorphism,
)


@dataclass(frozen=True, slots=True)
class FactorAnalysis:
    factor_length: int
    factors: tuple[tuple[str, ...], ...]
    occurrences: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class PeriodAnalysis:
    periods: tuple[int, ...]
    least_period: int
    primitive: bool


def factors_of_length(word: FiniteWord, factor_length: int) -> FactorAnalysis:
    if not 0 <= factor_length <= len(word.letters):
        raise ValueError("factor length must be between zero and the word length")
    positions: dict[tuple[str, ...], list[int]] = {}
    for start in range(len(word.letters) - factor_length + 1):
        factor = word.letters[start : start + factor_length]
        positions.setdefault(factor, []).append(start)
    factors = tuple(positions)
    return FactorAnalysis(
        factor_length=factor_length,
        factors=factors,
        occurrences=tuple(tuple(positions[factor]) for factor in factors),
    )


def factor_occurrences(word: FiniteWord, pattern: tuple[str, ...]) -> tuple[int, ...]:
    if any(letter not in word.alphabet for letter in pattern):
        raise ValueError("pattern letter is outside the declared alphabet")
    if not pattern:
        return tuple(range(len(word.letters) + 1))
    return tuple(
        start
        for start in range(len(word.letters) - len(pattern) + 1)
        if word.letters[start : start + len(pattern)] == pattern
    )


def periods(word: FiniteWord) -> PeriodAnalysis:
    length = len(word.letters)
    if length == 0:
        return PeriodAnalysis(periods=(), least_period=0, primitive=False)
    values = tuple(
        period
        for period in range(1, length + 1)
        if all(
            word.letters[index] == word.letters[index + period]
            for index in range(length - period)
        )
    )
    _, exponent = primitive_root(word)
    return PeriodAnalysis(
        periods=values,
        least_period=values[0],
        primitive=exponent == 1,
    )


def primitive_root(word: FiniteWord) -> tuple[tuple[str, ...], int]:
    length = len(word.letters)
    if length == 0:
        return ((), 1)
    for root_length in range(1, length + 1):
        if length % root_length == 0:
            root = word.letters[:root_length]
            if root * (length // root_length) == word.letters:
                return (root, length // root_length)
    raise RuntimeError("finite word did not admit itself as a primitive root")


def conjugates(word: FiniteWord) -> tuple[tuple[str, ...], ...]:
    if not word.letters:
        return ((),)
    rotations = {
        word.letters[index:] + word.letters[:index]
        for index in range(len(word.letters))
    }
    rank = {symbol: index for index, symbol in enumerate(word.alphabet)}
    return tuple(
        sorted(rotations, key=lambda value: tuple(rank[item] for item in value))
    )


def parikh_vector(word: FiniteWord) -> tuple[int, ...]:
    return tuple(word.letters.count(symbol) for symbol in word.alphabet)


def prefix_function(word: FiniteWord) -> tuple[int, ...]:
    result = [0] * len(word.letters)
    for index in range(1, len(word.letters)):
        border = result[index - 1]
        while border and word.letters[index] != word.letters[border]:
            border = result[border - 1]
        if word.letters[index] == word.letters[border]:
            border += 1
        result[index] = border
    return tuple(result)


def apply_morphism(morphism: WordMorphism, word: FiniteWord) -> FiniteWord:
    if word.alphabet != morphism.source_alphabet:
        raise ValueError("word alphabet must equal the morphism source alphabet")
    image_map = dict(zip(morphism.source_alphabet, morphism.images, strict=True))
    output_length = sum(len(image_map[letter]) for letter in word.letters)
    if output_length > MAX_MORPHISM_OUTPUT_LENGTH:
        raise ValueError("morphism output exceeds the length bound")
    letters = tuple(output for letter in word.letters for output in image_map[letter])
    return FiniteWord(alphabet=morphism.target_alphabet, letters=letters)


def compose_morphisms(first: WordMorphism, second: WordMorphism) -> WordMorphism:
    if first.target_alphabet != second.source_alphabet:
        raise ValueError("first target alphabet must equal second source alphabet")
    second_map = dict(zip(second.source_alphabet, second.images, strict=True))
    images = tuple(
        tuple(output for letter in image for output in second_map[letter])
        for image in first.images
    )
    if any(len(image) > MAX_MORPHISM_OUTPUT_LENGTH for image in images):
        raise ValueError("composed morphism image exceeds the length bound")
    return WordMorphism(
        source_alphabet=first.source_alphabet,
        target_alphabet=second.target_alphabet,
        images=images,
    )


def incidence_matrix(morphism: WordMorphism) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(image.count(target) for image in morphism.images)
        for target in morphism.target_alphabet
    )


__all__ = [
    "FactorAnalysis",
    "PeriodAnalysis",
    "apply_morphism",
    "compose_morphisms",
    "conjugates",
    "factor_occurrences",
    "factors_of_length",
    "incidence_matrix",
    "parikh_vector",
    "periods",
    "prefix_function",
    "primitive_root",
]
