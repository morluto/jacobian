"""Typed wire contracts for exact regular language operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.regular_languages.values import (
    DFA,
    MAX_COUNT_WORD_LENGTH,
    MAX_DFA_STATES,
    MAX_LABELED_AUTOMATON_STATES,
    MAX_TRANSITION_PROFILE_PATH_LENGTH,
    MAX_WORD_LENGTH,
    FiniteLabeledAutomaton,
)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"regular_language.{reason}", message)


class RunRequest(StrictModel):
    """Check if a word is accepted by a DFA."""

    dfa: DFA
    word: tuple[int, ...] = Field(max_length=MAX_WORD_LENGTH)

    @model_validator(mode="after")
    def require_valid_word(self) -> Self:
        for symbol in self.word:
            if not 0 <= symbol < self.dfa.alphabet_size:
                raise _validation_error(
                    "word_symbol_out_of_range",
                    "word symbols must be in 0..alphabet_size-1",
                )
        return self


class CountRequest(StrictModel):
    """Count accepted words of a given length."""

    dfa: DFA
    word_length: int = Field(ge=0, le=MAX_COUNT_WORD_LENGTH)


class ComplementRequest(StrictModel):
    """Compute the complement of a DFA's language."""

    dfa: DFA


class TransitionParikhProfileRequest(StrictModel):
    """Compute a complete transition-use profile for exact automaton paths.

    The automaton's ordered ``transition_id`` axis is authoritative. Admission
    derives path-extension, sparse-DP-cell, vector-coordinate, multiplicity,
    profile-cell, and serialized-result bounds before running the recurrence.
    """

    automaton: FiniteLabeledAutomaton = Field(
        description=(
            "Finite labeled transition carrier whose contiguous transition_id "
            "order is the complete profile coordinate axis."
        )
    )
    source_state: int = Field(
        ge=0,
        le=MAX_LABELED_AUTOMATON_STATES - 1,
        description="Path source on the automaton's zero-based state axis.",
    )
    target_state: int = Field(
        ge=0,
        le=MAX_LABELED_AUTOMATON_STATES - 1,
        description="Path target on the automaton's zero-based state axis.",
    )
    path_length: int = Field(
        ge=0,
        le=MAX_TRANSITION_PROFILE_PATH_LENGTH,
        description="Exact nonnegative number of transitions in every path.",
    )

    @model_validator(mode="after")
    def require_complete_bounded_profile(self) -> Self:
        from jacobian.math.regular_languages.operations import (
            _require_transition_profile_envelope,
        )

        try:
            _require_transition_profile_envelope(
                self.automaton,
                self.source_state,
                self.target_state,
                self.path_length,
            )
        except ValueError as exc:
            raise _validation_error("profile_admission_rejected", str(exc)) from exc
        return self


class RunResult(RunRequest):
    """Whether a word was accepted and the final state reached."""

    accepted: bool
    final_state: int = Field(ge=0, le=MAX_DFA_STATES - 1)
    state_trace: tuple[int, ...]
    method: Literal["DFA_SIMULATION"] = "DFA_SIMULATION"

    @model_validator(mode="after")
    def bind_run(self) -> Self:
        transitions = {
            (item.source, item.symbol): item.target for item in self.dfa.transitions
        }
        trace = [self.dfa.initial_state]
        for symbol in self.word:
            trace.append(transitions[(trace[-1], symbol)])
        if self.state_trace != tuple(trace) or self.final_state != trace[-1]:
            raise _validation_error(
                "run_trace_not_bound", "DFA run trace is not bound to its source"
            )
        if self.accepted != (self.final_state in self.dfa.accepting_states):
            raise _validation_error(
                "acceptance_mismatch", "DFA acceptance must agree with the final state"
            )
        return self


class CountResult(CountRequest):
    """Exact count of accepted words of a given length."""

    count: CanonicalInteger
    word_length: int = Field(ge=0, le=MAX_COUNT_WORD_LENGTH)
    method: Literal["MATRIX_POWERING"] = "MATRIX_POWERING"

    @model_validator(mode="after")
    def bind_count(self) -> Self:
        from jacobian.math.regular_languages.operations import count_accepted_words

        if int(self.count) != count_accepted_words(self.dfa, self.word_length):
            raise _validation_error(
                "count_not_bound", "word count is not bound to its DFA"
            )
        return self


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
    "TransitionParikhProfileRequest",
]
