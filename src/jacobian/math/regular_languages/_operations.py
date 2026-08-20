"""Domain adapter for regular language operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.regular_languages import (
    count_accepted_words,
    dfa_complement,
    dfa_run,
)
from jacobian.math.regular_languages._models import (
    ComplementRequest,
    ComplementResult,
    CountRequest,
    CountResult,
    RunRequest,
    RunResult,
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


__all__ = ["compute_complement", "compute_count", "compute_run"]
