"""Exact DFA kernels backed by integer matrix powering."""

from __future__ import annotations

from typing import Any

__all__ = ["count_accepted_words", "dfa_complement", "dfa_run"]


def _transition_map(dfa: Any) -> dict[tuple[int, int], int]:
    return {(tr.source, tr.symbol): tr.target for tr in dfa.transitions}


def _mat_mul(a: list[list[int]], b: list[list[int]], n: int) -> list[list[int]]:
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]
    return result


def _mat_pow(m: list[list[int]], p: int, n: int) -> list[list[int]]:
    if p == 0:
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    if p == 1:
        return m
    half = _mat_pow(m, p // 2, n)
    result = _mat_mul(half, half, n)
    if p % 2 == 1:
        result = _mat_mul(result, m, n)
    return result


def _build_matrix(dfa: Any, n: int) -> list[list[int]]:
    trans = _transition_map(dfa)
    matrix = [[0] * n for _ in range(n)]
    for (src, _symbol), dst in trans.items():
        matrix[src][dst] += 1
    return matrix


def _count_from_powered(powered: list[list[int]], dfa: Any, n: int) -> int:
    count = 0
    accepting = set(dfa.accepting_states)
    for target in range(n):
        if target in accepting:
            count += powered[dfa.initial_state][target]
    return count


def dfa_run(dfa: Any, word: tuple[int, ...]) -> tuple[bool, int]:
    """Simulate a DFA on a word; return (accepted, final_state)."""
    trans = _transition_map(dfa)
    state = dfa.initial_state
    for symbol in word:
        state = trans.get((state, symbol), state)
    accepted = state in set(dfa.accepting_states)
    return (accepted, state)


def count_accepted_words(dfa: Any, word_length: int) -> int:
    """Count accepted words of exact length via matrix powering."""
    n = dfa.state_count
    accepting = set(dfa.accepting_states)
    if word_length == 0:
        return 1 if dfa.initial_state in accepting else 0
    matrix = _build_matrix(dfa, n)
    powered = _mat_pow(matrix, word_length, n)
    return _count_from_powered(powered, dfa, n)


def dfa_complement(dfa: Any) -> Any:
    """Compute the complement DFA by flipping accepting states."""
    from jacobian.contracts.regular_languages import DFA

    all_states = set(range(dfa.state_count))
    new_accepting = tuple(sorted(all_states - set(dfa.accepting_states)))
    return DFA(
        state_count=dfa.state_count,
        alphabet_size=dfa.alphabet_size,
        transitions=dfa.transitions,
        initial_state=dfa.initial_state,
        accepting_states=new_accepting,
    )
