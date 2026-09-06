"""Provider-independent bounded values for symbolic dynamics."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_ALPHABET_SIZE = 16
MAX_SYMBOL_LENGTH = 64
MAX_FORBIDDEN_BLOCKS = 100
MAX_FORBIDDEN_BLOCK_LENGTH = 20
MAX_ADJACENCY_STATES = 50
MAX_ADJACENCY_ENTRY = 1_000_000
MAX_ENUMERATED_BLOCKS = 100_000
MAX_PRESENTATION_CELLS = MAX_ENUMERATED_BLOCKS
MAX_PRESENTATION_TRANSITIONS = MAX_PRESENTATION_CELLS
# Three complete integer vectors are returned. Even one-digit counts consume
# three digits per period under the aggregate profile budget.
MAX_PERIOD = 100_000 // 3

Symbol = Annotated[str, Field(min_length=1, max_length=MAX_SYMBOL_LENGTH)]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"symbolic_dynamics.{reason}", message)


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
            raise _validation_error(
                "alphabet_symbols_not_distinct", "alphabet symbols must be distinct"
            )
        for block in self.forbidden_blocks:
            if len(block) > MAX_FORBIDDEN_BLOCK_LENGTH:
                raise _validation_error(
                    "forbidden_block_length_bound",
                    "forbidden block exceeds the length bound",
                )
            if any(symbol not in self.alphabet for symbol in block):
                raise _validation_error(
                    "forbidden_block_symbol_outside_alphabet",
                    "forbidden block uses a symbol outside the alphabet",
                )
        return self


class AdjacencyShift(StrictModel):
    """An edge-shift carrier with nonnegative edge multiplicities."""

    matrix: tuple[
        Annotated[tuple[StrictInt, ...], Field(max_length=MAX_ADJACENCY_STATES)], ...
    ] = Field(max_length=MAX_ADJACENCY_STATES)
    two_sided: bool = True

    @model_validator(mode="after")
    def require_square_nonnegative_bounded_matrix(self) -> Self:
        size = len(self.matrix)
        if any(len(row) != size for row in self.matrix):
            raise _validation_error(
                "adjacency_matrix_not_square", "adjacency matrix must be square"
            )
        if any(
            entry < 0 or entry > MAX_ADJACENCY_ENTRY
            for row in self.matrix
            for entry in row
        ):
            raise _validation_error(
                "adjacency_entry_bound",
                "adjacency entries must be within the supported bounds",
            )
        return self


class LabeledTransition(StrictModel):
    source: StrictInt = Field(ge=0, le=MAX_ADJACENCY_STATES - 1)
    target: StrictInt = Field(ge=0, le=MAX_ADJACENCY_STATES - 1)
    appended_symbol: Symbol


_Block = Annotated[tuple[Symbol, ...], Field(max_length=MAX_FORBIDDEN_BLOCK_LENGTH)]
_AdjacencyRow = Annotated[tuple[StrictInt, ...], Field(max_length=MAX_ADJACENCY_STATES)]


class BlockPresentation(StrictModel):
    """A finite labeled overlap presentation."""

    alphabet: tuple[Symbol, ...] = Field(min_length=1, max_length=MAX_ALPHABET_SIZE)
    memory: StrictInt = Field(ge=0, le=MAX_FORBIDDEN_BLOCK_LENGTH)
    state_blocks: tuple[_Block, ...] = Field(max_length=MAX_ADJACENCY_STATES)
    forbidden_blocks: tuple[_Block, ...] = Field(
        default=(), max_length=MAX_FORBIDDEN_BLOCKS
    )
    transitions: tuple[LabeledTransition, ...] = Field(
        max_length=MAX_PRESENTATION_CELLS
    )
    adjacency_matrix: tuple[_AdjacencyRow, ...] = Field(max_length=MAX_ADJACENCY_STATES)
    two_sided: bool

    @model_validator(mode="after")
    def require_bound_presentation(self) -> Self:
        size = len(self.state_blocks)
        if len(self.adjacency_matrix) != size or any(
            len(row) != size for row in self.adjacency_matrix
        ):
            raise _validation_error(
                "presentation_adjacency_shape",
                "presentation adjacency must match its state blocks",
            )
        if any(
            entry < 0 or entry > MAX_ADJACENCY_ENTRY
            for row in self.adjacency_matrix
            for entry in row
        ):
            raise _validation_error(
                "presentation_adjacency_entry_bound",
                "presentation adjacency entries must be within the supported bounds",
            )
        if any(len(block) != self.memory for block in self.state_blocks):
            raise _validation_error(
                "presentation_state_block_memory",
                "presentation state blocks must match its memory",
            )
        if any(
            symbol not in self.alphabet
            for block in self.state_blocks
            for symbol in block
        ):
            raise _validation_error(
                "presentation_state_block_symbol_outside_alphabet",
                "presentation state blocks must use the presentation alphabet",
            )
        if any(
            symbol not in self.alphabet
            for block in self.forbidden_blocks
            for symbol in block
        ):
            raise _validation_error(
                "presentation_forbidden_symbol_outside_alphabet",
                "presentation forbidden blocks must use the presentation alphabet",
            )
        if any(
            transition.source >= size
            or transition.target >= size
            or transition.appended_symbol not in self.alphabet
            for transition in self.transitions
        ):
            raise _validation_error(
                "presentation_transition_outside_carrier",
                "presentation transition is outside its carrier",
            )
        return self


__all__ = [
    "MAX_ADJACENCY_ENTRY",
    "MAX_ADJACENCY_STATES",
    "MAX_ALPHABET_SIZE",
    "MAX_ENUMERATED_BLOCKS",
    "MAX_FORBIDDEN_BLOCKS",
    "MAX_FORBIDDEN_BLOCK_LENGTH",
    "MAX_PERIOD",
    "MAX_PRESENTATION_CELLS",
    "MAX_PRESENTATION_TRANSITIONS",
    "MAX_SYMBOL_LENGTH",
    "AdjacencyShift",
    "BlockPresentation",
    "ForbiddenBlockShift",
    "LabeledTransition",
    "Symbol",
]
