"""Typed wire contracts for exact regular language operations.

Domain-owned models (``DFA``, ``DFATransition``) live in
``jacobian.domains.regular_languages.contracts``; this module re-exports
them alongside the request/result contracts for wire-level consumers.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.domains.regular_languages.contracts import (
    DFA,
    MAX_DFA_STATES,
    MAX_WORD_LENGTH,
    DFATransition,
)


class RunRequest(ContractModel):
    """Check if a word is accepted by a DFA."""

    dfa: DFA
    word: tuple[int, ...] = Field(max_length=MAX_WORD_LENGTH)

    @model_validator(mode="after")
    def require_valid_word(self) -> Self:
        for symbol in self.word:
            if not (0 <= symbol < self.dfa.alphabet_size):
                raise ValueError("word symbols must be in 0..alphabet_size-1")
        return self


class CountRequest(ContractModel):
    """Count accepted words of a given length."""

    dfa: DFA
    word_length: int = Field(ge=0, le=200)


class ComplementRequest(ContractModel):
    """Compute the complement of a DFA's language."""

    dfa: DFA


class RunResult(ContractModel):
    """Whether a word was accepted."""

    accepted: bool
    final_state: int = Field(ge=0, le=MAX_DFA_STATES - 1)
    method: Literal["DFA_SIMULATION"] = "DFA_SIMULATION"


class CountResult(ContractModel):
    """Count of accepted words of a given length."""

    count: int = Field(ge=0)
    word_length: int = Field(ge=0, le=200)
    method: Literal["MATRIX_POWERING"] = "MATRIX_POWERING"


class ComplementResult(ContractModel):
    """The complement DFA."""

    dfa: DFA
    method: Literal["ACCEPTING_FLIP"] = "ACCEPTING_FLIP"


__all__ = [
    "DFA",
    "ComplementRequest",
    "ComplementResult",
    "CountRequest",
    "CountResult",
    "DFATransition",
    "RunRequest",
    "RunResult",
]
