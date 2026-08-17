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
    return RunResult(accepted=accepted, final_state=final_state)


def compute_count(request: CountRequest) -> CountResult:
    count = count_accepted_words(request.dfa, request.word_length)
    return CountResult(
        count=format_canonical_integer(count),
        word_length=request.word_length,
    )


def compute_complement(request: ComplementRequest) -> ComplementResult:
    return ComplementResult(dfa=dfa_complement(request.dfa))


__all__ = ["compute_complement", "compute_count", "compute_run"]
