"""Exact bounded maps between finite presentation carriers."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.edge_paths._models import (
    FiniteGroupPresentation,
    FiniteGroupWord,
    WordLetter,
)


class DirectRelatorMatchRequest(StrictModel):
    """Check only the finite direct-relator witness, not normal-closure membership."""

    source: FiniteGroupPresentation
    target: FiniteGroupPresentation
    generator_images: tuple[FiniteGroupWord, ...] = Field(min_length=0, max_length=64)


class DirectRelatorMatchResult(StrictModel):
    source: FiniteGroupPresentation
    target: FiniteGroupPresentation
    generator_images: tuple[FiniteGroupWord, ...]
    relator_images: tuple[FiniteGroupWord, ...]
    direct_relators_matched: bool
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


def direct_relator_match(
    source: FiniteGroupPresentation,
    target: FiniteGroupPresentation,
    generator_images: tuple[FiniteGroupWord, ...],
) -> DirectRelatorMatchResult:
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
    target_relators = {word.letters for word in target.relators} | {
        _inverse(word) for word in target.relators
    }
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
        if relator_image.letters and relator_image.letters not in target_relators:
            return DirectRelatorMatchResult(
                source=source,
                target=target,
                generator_images=generator_images,
                relator_images=tuple(images),
                direct_relators_matched=False,
                obstruction_index=i,
            )
    return DirectRelatorMatchResult(
        source=source,
        target=target,
        generator_images=generator_images,
        relator_images=tuple(images),
        direct_relators_matched=True,
    )


__all__ = [
    "DirectRelatorMatchRequest",
    "DirectRelatorMatchResult",
    "direct_relator_match",
]
