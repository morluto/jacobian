"""Domain-owned contract models for exact regular language operations."""

from __future__ import annotations

from typing import Self

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
    """One deterministic finite automaton.

    A valid DFA must have a transition for every (state, symbol) pair so that
    the automaton is total (no missing transitions).
    """

    state_count: int = Field(ge=1, le=MAX_DFA_STATES)
    alphabet_size: int = Field(ge=1, le=MAX_DFA_ALPHABET)
    transitions: tuple[DFATransition, ...] = Field(
        min_length=0, max_length=MAX_DFA_TRANSITIONS
    )
    initial_state: int = Field(ge=0, le=MAX_DFA_STATES - 1)
    accepting_states: tuple[int, ...] = Field(min_length=0, max_length=MAX_DFA_STATES)

    @model_validator(mode="after")
    def require_valid_dfa(self) -> Self:
        self._validate_states()
        self._validate_transitions()
        return self

    def _validate_states(self) -> None:
        if not (0 <= self.initial_state < self.state_count):
            raise ValueError("initial_state must be in 0..state_count-1")
        for state in self.accepting_states:
            if not (0 <= state < self.state_count):
                raise ValueError("accepting states must be in 0..state_count-1")
        if len(set(self.accepting_states)) != len(self.accepting_states):
            raise ValueError("accepting states must be unique")

    def _validate_transitions(self) -> None:
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
        expected = self.state_count * self.alphabet_size
        if len(seen) != expected:
            raise ValueError(
                f"DFA must be total: expected {expected} transitions for "
                f"{self.state_count} states x {self.alphabet_size} symbols, "
                f"got {len(seen)}"
            )
