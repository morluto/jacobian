"""Typed wire contracts for exact regular language operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.regular_languages.values import (
    DFA,
    MAX_COUNT_WORD_LENGTH,
    MAX_DFA_STATES,
    MAX_WORD_LENGTH,
)


class RunRequest(StrictModel):
    """Check if a word is accepted by a DFA."""

    dfa: DFA
    word: tuple[int, ...] = Field(max_length=MAX_WORD_LENGTH)

    @model_validator(mode="after")
    def require_valid_word(self) -> Self:
        for symbol in self.word:
            if not 0 <= symbol < self.dfa.alphabet_size:
                raise ValueError("word symbols must be in 0..alphabet_size-1")
        return self


class CountRequest(StrictModel):
    """Count accepted words of a given length."""

    dfa: DFA
    word_length: int = Field(ge=0, le=MAX_COUNT_WORD_LENGTH)


class ComplementRequest(StrictModel):
    """Compute the complement of a DFA's language."""

    dfa: DFA


class RunResult(StrictModel):
    """Whether a word was accepted and the final state reached."""

    accepted: bool
    final_state: int = Field(ge=0, le=MAX_DFA_STATES - 1)
    method: Literal["DFA_SIMULATION"] = "DFA_SIMULATION"


class CountResult(StrictModel):
    """Exact count of accepted words of a given length."""

    count: CanonicalInteger
    word_length: int = Field(ge=0, le=MAX_COUNT_WORD_LENGTH)
    method: Literal["MATRIX_POWERING"] = "MATRIX_POWERING"


class ComplementResult(StrictModel):
    """The complement DFA."""

    dfa: DFA
    method: Literal["ACCEPTING_FLIP"] = "ACCEPTING_FLIP"


__all__ = [
    "ComplementRequest",
    "ComplementResult",
    "CountRequest",
    "CountResult",
    "RunRequest",
    "RunResult",
]
