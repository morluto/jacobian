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
    return RunResult._from_kernel(
        request,
        accepted=accepted,
        final_state=final_state,
        state_trace=tuple(trace),
    )


def compute_count(request: CountRequest) -> CountResult:
    count = count_accepted_words(request.dfa, request.word_length)
    return CountResult._from_kernel(
        request,
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


def verify_run_result(result: RunResult) -> bool:
    """Verify an independently supplied DFA-run claim in its admitted envelope."""

    accepted, final_state = dfa_run(result.dfa, result.word)
    transitions = {
        (item.source, item.symbol): item.target for item in result.dfa.transitions
    }
    trace = [result.dfa.initial_state]
    for symbol in result.word:
        trace.append(transitions[(trace[-1], symbol)])
    return (
        result.accepted == accepted
        and result.final_state == final_state
        and result.state_trace == tuple(trace)
    )


def verify_count_result(result: CountResult) -> bool:
    """Verify an independently supplied exact count in its admitted envelope."""

    return int(result.count) == count_accepted_words(result.dfa, result.word_length)


def verify_transition_parikh_profile(
    profile: TransitionParikhProfile,
) -> bool:
    """Verify one independently supplied transition-Parikh profile claim."""

    from jacobian.math.regular_languages.operations import (
        _transition_parikh_profile_data,
    )

    expected_entries, expected_total = _transition_parikh_profile_data(
        profile.automaton,
        profile.source_state,
        profile.target_state,
        profile.path_length,
    )
    actual_entries = tuple(
        (entry.transition_counts, int(entry.multiplicity)) for entry in profile.entries
    )
    return (
        actual_entries == expected_entries
        and int(profile.total_path_count) == expected_total
    )


__all__ = [
    "compute_complement",
    "compute_count",
    "compute_run",
    "compute_transition_parikh_profile",
    "verify_count_result",
    "verify_run_result",
    "verify_transition_parikh_profile",
]
