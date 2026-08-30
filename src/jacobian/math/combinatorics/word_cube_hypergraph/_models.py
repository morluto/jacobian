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
    vertices = alphabet_size**dimension
    patterns = (alphabet_size + 1) ** dimension - vertices
    if vertices > 256:
        raise ValueError("word cube exceeds the 256-vertex hypergraph carrier")
    if patterns > 12_000 or patterns * alphabet_size > 36_000:
        raise ValueError("word cube exceeds the hyperedge or incidence envelope")
    label_bytes = 2 + dimension * (len(str(alphabet_size - 1)) + 1)
    if label_bytes > 64:
        raise ValueError("word-cube vertex labels exceed the 64-byte label envelope")


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
