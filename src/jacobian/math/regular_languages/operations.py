"""Exact regular language kernels backed by SymPy matrix powering."""

from __future__ import annotations

from jacobian.math.regular_languages.values import DFA

__all__ = ["count_accepted_words", "dfa_complement", "dfa_run"]


def _transition_map(dfa: DFA) -> dict[tuple[int, int], int]:
    return {(tr.source, tr.symbol): tr.target for tr in dfa.transitions}


def dfa_run(dfa: DFA, word: tuple[int, ...]) -> tuple[bool, int]:
    """Simulate a total DFA on a word; return ``(accepted, final_state)``."""

    transitions = _transition_map(dfa)
    state = dfa.initial_state
    for symbol in word:
        state = transitions[(state, symbol)]
    return (state in dfa.accepting_states, state)


def count_accepted_words(dfa: DFA, word_length: int) -> int:
    """Count accepted words of exact length via exact integer matrix powering."""

    import sympy

    state_count = dfa.state_count
    if word_length == 0:
        return 1 if dfa.initial_state in dfa.accepting_states else 0
    matrix = sympy.zeros(state_count, state_count)
    for (source, _symbol), target in _transition_map(dfa).items():
        matrix[source, target] += 1
    powered = matrix**word_length
    total = 0
    for target in dfa.accepting_states:
        total += int(powered[dfa.initial_state, target])
    return total


def dfa_complement(dfa: DFA) -> DFA:
    """Compute the complement DFA by flipping the accepting states."""

    accepting = set(dfa.accepting_states)
    return DFA(
        state_count=dfa.state_count,
        alphabet_size=dfa.alphabet_size,
        transitions=dfa.transitions,
        initial_state=dfa.initial_state,
        accepting_states=tuple(sorted(set(range(dfa.state_count)) - accepting)),
    )
