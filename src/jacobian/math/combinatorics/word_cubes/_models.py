"""Typed contracts for word-cube combinatorial-line hypergraphs."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_VERTICES,
    FiniteHypergraph,
)

MAX_ALPHABET_SIZE = MAX_VERTICES
MAX_DIMENSION = MAX_VERTICES.bit_length() - 1

Word = Annotated[tuple[int, ...], Field(min_length=1, max_length=MAX_DIMENSION)]


class CombinatorialLineHypergraphRequest(StrictModel):
    """Request to construct all Hales--Jewett lines of ``[q]^d``."""

    alphabet_size: int = Field(ge=2, le=MAX_ALPHABET_SIZE)
    dimension: int = Field(ge=1, le=MAX_DIMENSION)


class CombinatorialLine(StrictModel):
    """One line, with the unique wildcard pattern that generates it."""

    edge_id: str
    wildcard_positions: tuple[int, ...] = Field(min_length=1, max_length=MAX_DIMENSION)
    fixed_coordinates: tuple[tuple[int, int], ...] = Field(
        default=(), max_length=MAX_DIMENSION
    )
    vertices: tuple[Word, ...] = Field(min_length=2, max_length=MAX_ALPHABET_SIZE)


class CombinatorialLineHypergraphResult(StrictModel):
    """The complete word cube, its lines, and its generic hypergraph carrier."""

    alphabet_size: int
    dimension: int
    words: tuple[Word, ...] = Field(max_length=MAX_VERTICES)
    lines: tuple[CombinatorialLine, ...] = Field(max_length=MAX_EDGES)
    hypergraph: FiniteHypergraph


__all__ = [
    "MAX_ALPHABET_SIZE",
    "MAX_DIMENSION",
    "CombinatorialLine",
    "CombinatorialLineHypergraphRequest",
    "CombinatorialLineHypergraphResult",
    "Word",
]
