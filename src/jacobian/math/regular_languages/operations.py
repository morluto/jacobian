"""Exact finite-automaton and regular-language kernels."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.regular_languages._profile_admission import (
    require_transition_profile_envelope,
)
from jacobian.math.regular_languages.values import (
    DFA,
    AutomatonTransition,
    FiniteLabeledAutomaton,
    TransitionParikhCell,
    TransitionParikhProfile,
)

__all__ = [
    "count_accepted_words",
    "dfa_complement",
    "dfa_run",
    "dfa_transition_carrier",
    "transition_parikh_profile",
]

# Compatibility for owner-local boundary evidence; admission owns its use.
_MAX_TRANSITION_PROFILE_RESULT_BYTES = 4 * 1024 * 1024


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
    return sum(
        int(powered[dfa.initial_state, target]) for target in dfa.accepting_states
    )


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


def dfa_transition_carrier(dfa: DFA) -> FiniteLabeledAutomaton:
    """Project a DFA onto a labeled carrier with a stable transition axis."""

    ordered = sorted(dfa.transitions, key=lambda item: (item.source, item.symbol))
    return FiniteLabeledAutomaton(
        state_count=dfa.state_count,
        alphabet_size=dfa.alphabet_size,
        transitions=tuple(
            AutomatonTransition(
                transition_id=transition_id,
                source=transition.source,
                symbol=transition.symbol,
                target=transition.target,
            )
            for transition_id, transition in enumerate(ordered)
        ),
    )


def _outgoing_transitions(
    automaton: FiniteLabeledAutomaton,
) -> tuple[tuple[AutomatonTransition, ...], ...]:
    outgoing: list[list[AutomatonTransition]] = [
        [] for _ in range(automaton.state_count)
    ]
    for transition in automaton.transitions:
        outgoing[transition.source].append(transition)
    return tuple(tuple(transitions) for transitions in outgoing)


def _transition_parikh_profile_data(
    automaton: FiniteLabeledAutomaton,
    source_state: int,
    target_state: int,
    path_length: int,
) -> tuple[tuple[tuple[tuple[int, ...], int], ...], int]:
    """Compute canonical profile entries and an independent total path count."""

    expected_path_count = require_transition_profile_envelope(
        automaton, source_state, target_state, path_length
    )
    transition_count = len(automaton.transitions)
    outgoing = _outgoing_transitions(automaton)
    zero_vector = tuple(0 for _ in range(transition_count))
    layer: dict[tuple[int, tuple[int, ...]], int] = {(source_state, zero_vector): 1}
    for _ in range(path_length):
        if not layer:
            break
        next_layer: dict[tuple[int, tuple[int, ...]], int] = {}
        for (state, vector), multiplicity in layer.items():
            for transition in outgoing[state]:
                transition_id = transition.transition_id
                updated = (
                    *vector[:transition_id],
                    vector[transition_id] + 1,
                    *vector[transition_id + 1 :],
                )
                key = (transition.target, updated)
                next_layer[key] = next_layer.get(key, 0) + multiplicity
        layer = next_layer
    target_entries = sorted(
        (vector, multiplicity)
        for (state, vector), multiplicity in layer.items()
        if state == target_state
    )
    total_path_count = sum(multiplicity for _, multiplicity in target_entries)
    if total_path_count != expected_path_count:
        raise RuntimeError(
            "transition-Parikh recurrence disagrees with independent path counting"
        )
    return tuple(target_entries), total_path_count


def transition_parikh_profile(
    automaton: FiniteLabeledAutomaton,
    source_state: int,
    target_state: int,
    path_length: int,
) -> TransitionParikhProfile:
    """Return the exact transition-use histogram for fixed-endpoint paths."""

    target_entries, total_path_count = _transition_parikh_profile_data(
        automaton, source_state, target_state, path_length
    )
    return TransitionParikhProfile._from_kernel(
        automaton=automaton,
        source_state=source_state,
        target_state=target_state,
        path_length=path_length,
        entries=tuple(
            TransitionParikhCell(
                transition_counts=vector,
                multiplicity=format_canonical_integer(multiplicity),
            )
            for vector, multiplicity in target_entries
        ),
        total_path_count=format_canonical_integer(total_path_count),
    )
