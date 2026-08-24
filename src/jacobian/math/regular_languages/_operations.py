"""Domain adapter for regular language operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.regular_languages import (
    TransitionParikhProfile,
    count_accepted_words,
    dfa_complement,
    dfa_run,
    transition_parikh_profile,
)
from jacobian.math.regular_languages._models import (
    ComplementRequest,
    ComplementResult,
    CountRequest,
    CountResult,
    RunRequest,
    RunResult,
    TransitionParikhProfileRequest,
)


def compute_run(request: RunRequest) -> RunResult:
    accepted, final_state = dfa_run(request.dfa, request.word)
    transitions = {
        (item.source, item.symbol): item.target for item in request.dfa.transitions
    }
    trace = [request.dfa.initial_state]
    for symbol in request.word:
        trace.append(transitions[(trace[-1], symbol)])
    return RunResult(
        **request.model_dump(),
        accepted=accepted,
        final_state=final_state,
        state_trace=tuple(trace),
    )


def compute_count(request: CountRequest) -> CountResult:
    count = count_accepted_words(request.dfa, request.word_length)
    return CountResult(
        **request.model_dump(),
        count=format_canonical_integer(count),
    )


def compute_complement(request: ComplementRequest) -> ComplementResult:
    return ComplementResult(dfa=dfa_complement(request.dfa))


def compute_transition_parikh_profile(
    request: TransitionParikhProfileRequest,
) -> TransitionParikhProfile:
    return transition_parikh_profile(
        request.automaton,
        request.source_state,
        request.target_state,
        request.path_length,
    )


__all__ = [
    "compute_complement",
    "compute_count",
    "compute_run",
    "compute_transition_parikh_profile",
]
