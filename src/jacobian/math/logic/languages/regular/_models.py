"""Typed wire contracts for exact regular language operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.logic.languages.regular.values import (
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


class RunResult(RunRequest):
    """Whether a word was accepted and the final state reached."""

    accepted: bool
    final_state: int = Field(ge=0, le=MAX_DFA_STATES - 1)
    state_trace: tuple[int, ...]
    method: Literal["DFA_SIMULATION"] = "DFA_SIMULATION"

    @model_validator(mode="after")
    def require_run_shape(self) -> Self:
        if len(self.state_trace) != len(self.word) + 1:
            raise _validation_error(
                "run_trace_length_mismatch",
                "DFA run trace must contain the initial state and one state per symbol",
            )
        if (
            self.state_trace[0] != self.dfa.initial_state
            or self.final_state != self.state_trace[-1]
        ):
            raise _validation_error(
                "run_trace_endpoint_mismatch",
                "DFA run trace must begin at the initial state and end at final_state",
            )
        if any(not 0 <= state < self.dfa.state_count for state in self.state_trace):
            raise _validation_error(
                "run_trace_state_out_of_range",
                "DFA run trace contains a state outside its carrier",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: RunRequest,
        *,
        accepted: bool,
        final_state: int,
        state_trace: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            dfa=request.dfa,
            word=request.word,
            accepted=accepted,
            final_state=final_state,
            state_trace=state_trace,
            method="DFA_SIMULATION",
        )


class CountResult(CountRequest):
    """Exact count of accepted words of a given length."""

    count: CanonicalInteger
    word_length: int = Field(ge=0, le=MAX_COUNT_WORD_LENGTH)
    method: Literal["MATRIX_POWERING"] = "MATRIX_POWERING"

    @model_validator(mode="after")
    def require_nonnegative_count(self) -> Self:
        if parse_canonical_integer(self.count) < 0:
            raise _validation_error("count_negative", "word count must be nonnegative")
        return self

    @classmethod
    def _from_kernel(cls, request: CountRequest, *, count: CanonicalInteger) -> Self:
        return cls.model_construct(
            dfa=request.dfa,
            word_length=request.word_length,
            count=count,
            method="MATRIX_POWERING",
        )


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
