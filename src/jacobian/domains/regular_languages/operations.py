"""Domain adapter for regular language operations."""

from __future__ import annotations

from jacobian.contracts.regular_languages import (
    ComplementRequest,
    ComplementResult,
    CountRequest,
    CountResult,
    RunRequest,
    RunResult,
)
from jacobian.math.regular_languages import (
    count_accepted_words,
    dfa_complement,
    dfa_run,
)


def compute_run(request: RunRequest) -> RunResult:
    accepted, final_state = dfa_run(request.dfa, request.word)
    return RunResult(accepted=accepted, final_state=final_state)


def compute_count(request: CountRequest) -> CountResult:
    count = count_accepted_words(request.dfa, request.word_length)
    return CountResult(count=count, word_length=request.word_length)


def compute_complement(request: ComplementRequest) -> ComplementResult:
    result = dfa_complement(request.dfa)
    return ComplementResult(dfa=result)
