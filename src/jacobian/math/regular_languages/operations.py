"""Exact DFA kernels backed by integer matrix powering via SymPy."""

from __future__ import annotations

from typing import Any

__all__ = ["count_accepted_words", "dfa_complement", "dfa_run"]


def _transition_map(dfa: Any) -> dict[tuple[int, int], int]:
    return {(tr.source, tr.symbol): tr.target for tr in dfa.transitions}


def _build_matrix(dfa: Any, n: int):
    """Build the adjacency matrix of the DFA transition graph."""
    import sympy

    trans = _transition_map(dfa)
    matrix = sympy.zeros(n, n)
    for (src, _symbol), dst in trans.items():
        matrix[src, dst] += 1
    return matrix


def _count_from_powered(powered, dfa: Any, n: int) -> int:
    count = 0
    accepting = set(dfa.accepting_states)
    for target in range(n):
        if target in accepting:
            count += int(powered[dfa.initial_state, target])
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
    """Count accepted words of exact length via SymPy matrix powering."""
    n = dfa.state_count
    accepting = set(dfa.accepting_states)
    if word_length == 0:
        return 1 if dfa.initial_state in accepting else 0
    matrix = _build_matrix(dfa, n)
    powered = matrix**word_length
    return _count_from_powered(powered, dfa, n)


def dfa_complement(dfa: Any) -> Any:
    """Compute the complement DFA by flipping accepting states."""
    from jacobian.domains.regular_languages.contracts import DFA

    all_states = set(range(dfa.state_count))
    new_accepting = tuple(sorted(all_states - set(dfa.accepting_states)))
    return DFA(
        state_count=dfa.state_count,
        alphabet_size=dfa.alphabet_size,
        transitions=dfa.transitions,
        initial_state=dfa.initial_state,
        accepting_states=new_accepting,
    )
