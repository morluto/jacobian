"""Exact bounded maps between finite presentation carriers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.edge_paths._models import (
    MAX_WORD,
    FiniteGroupPresentation,
    FiniteGroupWord,
    WordLetter,
)


class PresentationHomomorphismRequest(StrictModel):
    source: FiniteGroupPresentation
    target: FiniteGroupPresentation
    generator_images: tuple[FiniteGroupWord, ...] = Field(min_length=0, max_length=64)


class PresentationHomomorphismResult(StrictModel):
    source: FiniteGroupPresentation
    target: FiniteGroupPresentation
    generator_images: tuple[FiniteGroupWord, ...]
    relator_images: tuple[FiniteGroupWord, ...]
    relators_preserved: bool
    # A bounded normal-closure search can prove preservation, but failure to
    # find a van Kampen word is not a negative group-theoretic conclusion.
    normal_closure_status: Literal["PROVED", "UNKNOWN"] = "UNKNOWN"
    obstruction_index: int | None = None


def _reduce(letters: Any) -> tuple[WordLetter, ...]:
    out: list[WordLetter] = []
    for letter in letters:
        if (
            out
            and out[-1].generator == letter.generator
            and out[-1].exponent == -letter.exponent
        ):
            out.pop()
        else:
            out.append(letter)
    return tuple(out)


def _inverse(word: FiniteGroupWord) -> tuple[WordLetter, ...]:
    return tuple(
        WordLetter(generator=x.generator, exponent=-x.exponent)
        for x in reversed(word.letters)
    )


MAX_NORMAL_CLOSURE_FACTORS = 8


def _concat(*words: tuple[WordLetter, ...]) -> tuple[WordLetter, ...]:
    return _reduce(letter for word in words for letter in word)


def _normal_closure_proof(
    word: tuple[WordLetter, ...], target: FiniteGroupPresentation
) -> bool:
    """Boundedly prove membership by enumerating short conjugate products."""
    if not word:
        return True
    relators = tuple(
        (relator.letters, _inverse(relator))
        for relator in target.relators
    )
    if not relators:
        return False
    words: set[tuple[WordLetter, ...]] = {()}
    # Include bounded conjugators over the target generator alphabet. This is
    # complete for the admitted envelope only when a witness is found; a miss
    # is intentionally reported as UNKNOWN by the public operation.
    occurring = {
        letter.generator for pair in relators for relator in pair for letter in relator
    }
    alphabet = tuple(
        WordLetter(generator=i, exponent=e)
        for i in sorted(occurring)
        for e in (-1, 1)
    )
    conjugators = [(), *((letter,) for letter in alphabet)]
    factors = [
        _concat(conjugator, relator, _inverse(FiniteGroupWord(letters=conjugator)))
        for conjugator in conjugators
        for pair in relators
        for relator in pair
    ]
    for _ in range(MAX_NORMAL_CLOSURE_FACTORS):
        words = {
            _concat(existing, factor)
            for existing in words
            for factor in factors
            if len(_concat(existing, factor)) <= MAX_WORD
        }
        if len(words) > 100_000:
            return False
        if word in words:
            return True
    return False


def homomorphism(
    source: FiniteGroupPresentation,
    target: FiniteGroupPresentation,
    generator_images: tuple[FiniteGroupWord, ...],
) -> PresentationHomomorphismResult:
    if len(generator_images) != len(source.generators):
        raise OperationDomainValidationError(
            location=("generator_images",),
            code="presentation_map.generator_axis",
            message="one target word is required for every source generator",
        )
    if any(
        letter.generator >= len(target.generators)
        for word in generator_images
        for letter in word.letters
    ):
        raise OperationDomainValidationError(
            location=("generator_images",),
            code="presentation_map.target_generator",
            message="a generator image names no target generator",
        )
    images = []
    for relator in source.relators:
        expanded: list[WordLetter] = []
        for letter in relator.letters:
            image_word = generator_images[letter.generator].letters
            if letter.exponent == 1:
                expanded.extend(image_word)
            else:
                expanded.extend(_inverse(FiniteGroupWord(letters=image_word)))
        images.append(FiniteGroupWord(letters=_reduce(expanded)))
    for i, relator_image in enumerate(images):
        if not _normal_closure_proof(relator_image.letters, target):
            return PresentationHomomorphismResult(
                source=source,
                target=target,
                generator_images=generator_images,
                relator_images=tuple(images),
                relators_preserved=False,
                normal_closure_status="UNKNOWN",
                obstruction_index=i,
            )
    return PresentationHomomorphismResult(
        source=source,
        target=target,
        generator_images=generator_images,
        relator_images=tuple(images),
        relators_preserved=True,
        normal_closure_status="PROVED",
    )


__all__ = [
    "PresentationHomomorphismRequest",
    "PresentationHomomorphismResult",
    "homomorphism",
]
