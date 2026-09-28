"""Exact bounded subsequence operations on ordered finite words."""

from __future__ import annotations

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.algebraic._models import (
    MAX_LIS_DP_WORK,
    MAX_LIS_INDEX_DIGITS,
    MAX_LIS_OUTPUT_SCALARS,
    MAX_LIS_WORD_LENGTH,
    MAX_LIS_WORD_PAYLOAD_SCALARS,
    LongestDecreasingSubsequenceRequest,
    LongestDecreasingSubsequenceResult,
    LongestIncreasingSubsequenceRequest,
    LongestIncreasingSubsequenceResult,
)
from jacobian.math.combinatorics.algebraic._rsk import word_payload_scalars
from jacobian.math.logic.languages.words.values import FiniteWord


def longest_increasing_subsequence(
    word: FiniteWord | LongestIncreasingSubsequenceRequest,
) -> LongestIncreasingSubsequenceResult:
    """Return one deterministic strict longest increasing subsequence.

    Among equal-length witnesses, the dynamic program keeps the earliest
    predecessor at each endpoint, then the earliest endpoint. The complete
    search examines exactly at most ``n * (n - 1) // 2`` predecessor pairs.
    """

    if isinstance(word, LongestIncreasingSubsequenceRequest):
        word = word.word
    n = len(word.letters)
    if n > MAX_LIS_WORD_LENGTH:
        raise OperationResourceAdmissionError(
            location=("word", "letters"),
            code="algebraic_combinatorics.lis_word_length",
            message="word length exceeds the admitted strict-LIS envelope",
        )
    payload_scalars = word_payload_scalars(word)
    if payload_scalars > MAX_LIS_WORD_PAYLOAD_SCALARS:
        raise OperationResourceAdmissionError(
            location=("word",),
            code="algebraic_combinatorics.lis_word_bytes",
            message="word payload exceeds the admitted strict-LIS envelope",
        )
    work = n * (n - 1) // 2
    if work > MAX_LIS_DP_WORK:
        raise OperationResourceAdmissionError(
            location=("word", "letters"),
            code="algebraic_combinatorics.lis_dp_work",
            message="strict-LIS dynamic-programming work exceeds its admitted bound",
        )
    output_bound = 2 * payload_scalars + (n + 1) * MAX_LIS_INDEX_DIGITS
    if output_bound > MAX_LIS_OUTPUT_SCALARS:
        raise OperationResourceAdmissionError(
            location=("word",),
            code="algebraic_combinatorics.lis_output_bytes",
            message="strict-LIS result exceeds the admitted output-size envelope",
        )

    rank = {letter: index for index, letter in enumerate(word.alphabet)}
    lengths = [1] * n
    predecessors = [-1] * n
    best_end = -1
    for end in range(n):
        request_checkpoint("during strict-LIS dynamic programming")
        end_rank = rank[word.letters[end]]
        for previous in range(end):
            if rank[word.letters[previous]] < end_rank:
                candidate = lengths[previous] + 1
                if candidate > lengths[end]:
                    lengths[end] = candidate
                    predecessors[end] = previous
        if best_end < 0 or lengths[end] > lengths[best_end]:
            best_end = end

    indices: list[int] = []
    cursor = best_end
    while cursor >= 0:
        indices.append(cursor)
        cursor = predecessors[cursor]
    indices.reverse()
    chosen = tuple(indices)
    return LongestIncreasingSubsequenceResult.model_construct(
        source_word=word,
        length=len(chosen),
        indices=chosen,
        values=tuple(word.letters[index] for index in chosen),
    )


def longest_decreasing_subsequence(
    word: FiniteWord | LongestDecreasingSubsequenceRequest,
) -> LongestDecreasingSubsequenceResult:
    """Return one deterministic strict longest decreasing subsequence."""

    if isinstance(word, LongestDecreasingSubsequenceRequest):
        word = word.word
    n = len(word.letters)
    if n > MAX_LIS_WORD_LENGTH:
        raise OperationResourceAdmissionError(
            location=("word", "letters"),
            code="algebraic_combinatorics.lds_word_length",
            message="word length exceeds the admitted strict-LDS envelope",
        )
    payload_scalars = word_payload_scalars(word)
    if payload_scalars > MAX_LIS_WORD_PAYLOAD_SCALARS:
        raise OperationResourceAdmissionError(
            location=("word",),
            code="algebraic_combinatorics.lds_word_bytes",
            message="word payload exceeds the admitted strict-LDS envelope",
        )
    work = n * (n - 1) // 2
    if work > MAX_LIS_DP_WORK:
        raise OperationResourceAdmissionError(
            location=("word", "letters"),
            code="algebraic_combinatorics.lds_dp_work",
            message="strict-LDS dynamic-programming work exceeds its admitted bound",
        )
    output_bound = 2 * payload_scalars + (n + 1) * MAX_LIS_INDEX_DIGITS
    if output_bound > MAX_LIS_OUTPUT_SCALARS:
        raise OperationResourceAdmissionError(
            location=("word",),
            code="algebraic_combinatorics.lds_output_bytes",
            message="strict-LDS result exceeds the admitted output-size envelope",
        )

    rank = {letter: index for index, letter in enumerate(word.alphabet)}
    lengths = [1] * n
    predecessors = [-1] * n
    best_end = -1
    for end in range(n):
        request_checkpoint("during strict-LDS dynamic programming")
        end_rank = rank[word.letters[end]]
        for previous in range(end):
            if rank[word.letters[previous]] > end_rank:
                candidate = lengths[previous] + 1
                if candidate > lengths[end]:
                    lengths[end] = candidate
                    predecessors[end] = previous
        if best_end < 0 or lengths[end] > lengths[best_end]:
            best_end = end

    indices: list[int] = []
    cursor = best_end
    while cursor >= 0:
        indices.append(cursor)
        cursor = predecessors[cursor]
    indices.reverse()
    chosen = tuple(indices)
    return LongestDecreasingSubsequenceResult.model_construct(
        source_word=word,
        length=len(chosen),
        indices=chosen,
        values=tuple(word.letters[index] for index in chosen),
    )


__all__ = ["longest_decreasing_subsequence", "longest_increasing_subsequence"]
