"""Provider-independent values for exact regular languages over finite DFAs."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_DFA_STATES = 64
MAX_DFA_ALPHABET = 32
MAX_DFA_TRANSITIONS = 4096
MAX_WORD_LENGTH = 1000
MAX_COUNT_WORD_LENGTH = 200


class DFATransition(StrictModel):
    """One transition: from ``source`` on ``symbol`` to ``target``."""

    source: int = Field(ge=0, le=MAX_DFA_STATES - 1)
    symbol: int = Field(ge=0, le=MAX_DFA_ALPHABET - 1)
    target: int = Field(ge=0, le=MAX_DFA_STATES - 1)


class DFA(StrictModel):
    """One total deterministic finite automaton over an integer alphabet.

    A valid DFA declares exactly one transition for every ``(state, symbol)``
    pair so that simulation and word counting share one consistent semantics.
    """

    state_count: int = Field(ge=1, le=MAX_DFA_STATES)
    alphabet_size: int = Field(ge=1, le=MAX_DFA_ALPHABET)
    transitions: tuple[DFATransition, ...] = Field(
        min_length=0,
        max_length=MAX_DFA_TRANSITIONS,
    )
    initial_state: int = Field(ge=0, le=MAX_DFA_STATES - 1)
    accepting_states: tuple[int, ...] = Field(
        min_length=0,
        max_length=MAX_DFA_STATES,
    )

    @model_validator(mode="after")
    def require_total_deterministic_dfa(self) -> Self:
        if not 0 <= self.initial_state < self.state_count:
            raise ValueError("initial_state must be in 0..state_count-1")
        if any(not 0 <= state < self.state_count for state in self.accepting_states):
            raise ValueError("accepting states must be in 0..state_count-1")
        if len(set(self.accepting_states)) != len(self.accepting_states):
            raise ValueError("accepting states must be unique")
        seen: set[tuple[int, int]] = set()
        for transition in self.transitions:
            if not 0 <= transition.source < self.state_count:
                raise ValueError("transition source must be in 0..state_count-1")
            if not 0 <= transition.target < self.state_count:
                raise ValueError("transition target must be in 0..state_count-1")
            if not 0 <= transition.symbol < self.alphabet_size:
                raise ValueError("transition symbol must be in 0..alphabet_size-1")
            key = (transition.source, transition.symbol)
            if key in seen:
                raise ValueError(
                    "DFA must be deterministic (no duplicate source/symbol)"
                )
            seen.add(key)
        expected = self.state_count * self.alphabet_size
        if len(seen) != expected:
            raise ValueError(
                "DFA must be total: expected one transition for every "
                f"state-symbol pair ({expected}), got {len(seen)}"
            )
        return self


__all__ = ["DFA", "DFATransition"]
