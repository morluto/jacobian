"""Provider-independent bounded values for symbolic dynamics."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_ALPHABET_SIZE = 16
MAX_SYMBOL_LENGTH = 64
MAX_FORBIDDEN_BLOCKS = 100
MAX_FORBIDDEN_BLOCK_LENGTH = 20
MAX_ADJACENCY_STATES = 50
MAX_ADJACENCY_ENTRY = 1_000_000
MAX_ENUMERATED_BLOCKS = 100_000
MAX_PERIOD = 50

Symbol = Annotated[str, Field(min_length=1, max_length=MAX_SYMBOL_LENGTH)]


class ForbiddenBlockShift(StrictModel):
    """A finite-alphabet shift specified by forbidden contiguous blocks."""

    alphabet: tuple[Symbol, ...] = Field(min_length=1, max_length=MAX_ALPHABET_SIZE)
    forbidden_blocks: tuple[tuple[str, ...], ...] = Field(
        max_length=MAX_FORBIDDEN_BLOCKS
    )
    two_sided: bool = True

    @model_validator(mode="after")
    def require_bounded_words_over_distinct_alphabet(self) -> Self:
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet symbols must be distinct")
        for block in self.forbidden_blocks:
            if len(block) > MAX_FORBIDDEN_BLOCK_LENGTH:
                raise ValueError("forbidden block exceeds the length bound")
            if any(symbol not in self.alphabet for symbol in block):
                raise ValueError("forbidden block uses a symbol outside the alphabet")
        return self


class AdjacencyShift(StrictModel):
    """An edge-shift carrier with nonnegative edge multiplicities."""

    matrix: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_ADJACENCY_STATES
    )
    two_sided: bool = True

    @model_validator(mode="after")
    def require_square_nonnegative_bounded_matrix(self) -> Self:
        size = len(self.matrix)
        if any(len(row) != size for row in self.matrix):
            raise ValueError("adjacency matrix must be square")
        if any(
            entry < 0 or entry > MAX_ADJACENCY_ENTRY
            for row in self.matrix
            for entry in row
        ):
            raise ValueError("adjacency entries must be within the supported bounds")
        return self


class LabeledTransition(StrictModel):
    source: int = Field(ge=0)
    target: int = Field(ge=0)
    appended_symbol: str


class BlockPresentation(StrictModel):
    """A finite labeled overlap presentation."""

    alphabet: tuple[str, ...]
    memory: int = Field(ge=0)
    state_blocks: tuple[tuple[str, ...], ...]
    transitions: tuple[LabeledTransition, ...]
    adjacency_matrix: tuple[tuple[int, ...], ...]
    two_sided: bool

    @model_validator(mode="after")
    def require_bound_presentation(self) -> Self:
        size = len(self.state_blocks)
        if len(self.adjacency_matrix) != size or any(
            len(row) != size for row in self.adjacency_matrix
        ):
            raise ValueError("presentation adjacency must match its state blocks")
        if any(len(block) != self.memory for block in self.state_blocks):
            raise ValueError("presentation state blocks must match its memory")
        if any(
            transition.source >= size
            or transition.target >= size
            or transition.appended_symbol not in self.alphabet
            for transition in self.transitions
        ):
            raise ValueError("presentation transition is outside its carrier")
        counts = [[0] * size for _ in range(size)]
        for transition in self.transitions:
            counts[transition.source][transition.target] += 1
        if self.adjacency_matrix != tuple(tuple(row) for row in counts):
            raise ValueError("presentation adjacency does not count its transitions")
        return self


__all__ = [
    "MAX_ADJACENCY_ENTRY",
    "MAX_ADJACENCY_STATES",
    "MAX_ALPHABET_SIZE",
    "MAX_ENUMERATED_BLOCKS",
    "MAX_FORBIDDEN_BLOCKS",
    "MAX_FORBIDDEN_BLOCK_LENGTH",
    "MAX_PERIOD",
    "MAX_SYMBOL_LENGTH",
    "AdjacencyShift",
    "BlockPresentation",
    "ForbiddenBlockShift",
    "LabeledTransition",
    "Symbol",
]
