"""Affine-map word collision profile kernel."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

from jacobian._exact import CanonicalRational
from jacobian.math.algebra.affine_map_word_collision._models import (
    CollisionRow,
    WordCollisionProfileResult,
)

__all__ = ["compute_word_collision_profile"]


def compute_word_collision_profile(
    generators: tuple[tuple[Fraction, Fraction], ...],
    depth: int,
) -> WordCollisionProfileResult:
    """Compute the complete word collision profile of an affine-map family.

    For each generator word of length ``depth``, compose the corresponding
    affine maps and group words by their exact composed map.
    Convention: word (i_1,...,i_d) represents f_{i_d} o ... o f_{i_1}.
    """
    from jacobian.math.algebra.affine_map_word_collision._models import (
        AffineMapSpec,
    )

    gen_specs = [
        AffineMapSpec(
            slope=CanonicalRational.from_fraction(slope),
            intercept=CanonicalRational.from_fraction(intercept),
        )
        for slope, intercept in generators
    ]

    class_to_words: dict[tuple[Fraction, Fraction], list[tuple[int, ...]]] = {}

    for word in product(range(len(generators)), repeat=depth):
        a, b = _compose_word(generators, word)
        key = (a, b)
        if key not in class_to_words:
            class_to_words[key] = []
        class_to_words[key].append(word)

    rows: list[CollisionRow] = []
    for (slope, intercept), words in sorted(
        class_to_words.items(),
        key=lambda kv: (kv[0][0], kv[0][1]),
    ):
        rows.append(
            CollisionRow(
                slope=CanonicalRational.from_fraction(slope),
                intercept=CanonicalRational.from_fraction(intercept),
                multiplicity=len(words),
                words=tuple(sorted(words)),
            )
        )

    return WordCollisionProfileResult(
        generators=tuple(gen_specs),
        depth=depth,
        rows=tuple(rows),
    )


def _compose_word(
    generators: tuple[tuple[Fraction, Fraction], ...],
    word: tuple[int, ...],
) -> tuple[Fraction, Fraction]:
    """Compose affine maps according to the convention f_{i_d} o ... o f_{i_1}.

    Each generator is x -> a*x + b. Composition (A,B) o (C,D) = (A*C, A*D+B).
    """
    a = Fraction(1)
    b = Fraction(0)
    for idx in reversed(word):
        ga, gb = generators[idx]
        a, b = ga * a, ga * b + gb
    return a, b
