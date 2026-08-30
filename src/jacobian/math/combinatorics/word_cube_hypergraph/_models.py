"""Typed contracts for the word-cube combinatorial-line hypergraph."""

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)


def require_word_cube_envelope(alphabet_size: int, dimension: int) -> None:
    if alphabet_size < 1 or dimension < 1:
        raise ValueError("word cubes require positive alphabet_size and dimension")
    label_bytes = 2 + dimension * (len(str(alphabet_size - 1)) + 1)
    if label_bytes > 64:
        raise ValueError("word-cube vertex labels exceed the 64-byte label envelope")
    vertices = _bounded_power(alphabet_size, dimension, 256)
    if vertices > 256:
        raise ValueError("word cube exceeds the 256-vertex hypergraph carrier")
    augmented_words = _bounded_power(alphabet_size + 1, dimension, 12_000 + vertices)
    patterns = augmented_words - vertices
    if patterns > 12_000 or patterns * alphabet_size > 36_000:
        raise ValueError("word cube exceeds the hyperedge or incidence envelope")


def _bounded_power(base: int, exponent: int, limit: int) -> int:
    """Return ``base**exponent`` or ``limit + 1`` without oversized integers."""
    result = 1
    factor = base
    remaining = exponent
    while remaining:
        if remaining & 1:
            if factor > limit or result > limit // factor:
                return limit + 1
            result *= factor
        remaining >>= 1
        if remaining:
            if factor > limit or factor > limit // factor:
                factor = limit + 1
            else:
                factor *= factor
    return result


class WordCubeRequest(StrictModel):
    """Request to construct the combinatorial-line hypergraph of [q]^d."""

    alphabet_size: int = Field(ge=1)
    dimension: int = Field(ge=1)

    @model_validator(mode="after")
    def require_bounded_cube(self) -> Self:
        try:
            require_word_cube_envelope(self.alphabet_size, self.dimension)
        except ValueError as exc:
            raise PydanticCustomError("word_cube.envelope_exceeded", str(exc)) from exc
        return self


class WordCubeResult(StrictModel):
    """The combinatorial-line hypergraph of [q]^d."""

    alphabet_size: int
    dimension: int
    hypergraph: FiniteHypergraph


__all__ = ["WordCubeRequest", "WordCubeResult", "require_word_cube_envelope"]
