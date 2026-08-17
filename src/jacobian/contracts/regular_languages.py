"""Typed wire contracts for exact regular language operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel

MAX_DFA_STATES = 64
MAX_DFA_ALPHABET = 32
MAX_DFA_TRANSITIONS = 4096
MAX_WORD_LENGTH = 1000


class DFATransition(ContractModel):
    """One transition: from state on symbol to next state."""

    source: int = Field(ge=0, le=MAX_DFA_STATES - 1)
    symbol: int = Field(ge=0, le=MAX_DFA_ALPHABET - 1)
    target: int = Field(ge=0, le=MAX_DFA_STATES - 1)


class DFA(ContractModel):
    """One deterministic finite automaton."""

    state_count: int = Field(ge=1, le=MAX_DFA_STATES)
    alphabet_size: int = Field(ge=1, le=MAX_DFA_ALPHABET)
    transitions: tuple[DFATransition, ...] = Field(
        min_length=0, max_length=MAX_DFA_TRANSITIONS
    )
    initial_state: int = Field(ge=0, le=MAX_DFA_STATES - 1)
    accepting_states: tuple[int, ...] = Field(min_length=0, max_length=MAX_DFA_STATES)

    @model_validator(mode="after")
    def require_valid_dfa(self) -> Self:
        if not (0 <= self.initial_state < self.state_count):
            raise ValueError("initial_state must be in 0..state_count-1")
        for state in self.accepting_states:
            if not (0 <= state < self.state_count):
                raise ValueError("accepting states must be in 0..state_count-1")
        if len(set(self.accepting_states)) != len(self.accepting_states):
            raise ValueError("accepting states must be unique")
        seen: set[tuple[int, int]] = set()
        for tr in self.transitions:
            if not (0 <= tr.source < self.state_count):
                raise ValueError("transition source must be in 0..state_count-1")
            if not (0 <= tr.target < self.state_count):
                raise ValueError("transition target must be in 0..state_count-1")
            if not (0 <= tr.symbol < self.alphabet_size):
                raise ValueError("transition symbol must be in 0..alphabet_size-1")
            key = (tr.source, tr.symbol)
            if key in seen:
                raise ValueError(
                    "DFA must be deterministic (no duplicate source/symbol)"
                )
            seen.add(key)
        return self


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
