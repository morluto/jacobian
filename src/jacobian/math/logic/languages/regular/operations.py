"""Exact finite-automaton and regular-language kernels."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.languages.regular._profile_admission import (
    TransitionParikhAdmissionPlan,
    admit_transition_profile,
)
from jacobian.math.logic.languages.regular.values import (
    DFA,
    MAX_COUNT_MATRIX_WORK,
    MAX_COUNT_RESULT_DIGITS,
    MAX_COUNT_WORD_LENGTH,
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


def _transition_map(dfa: DFA) -> dict[tuple[int, int], int]:
    return {(tr.source, tr.symbol): tr.target for tr in dfa.transitions}


def dfa_run(dfa: DFA, word: tuple[int, ...]) -> tuple[bool, int]:
    """Simulate a total DFA on a word; return ``(accepted, final_state)``."""

    if any(not 0 <= symbol < dfa.alphabet_size for symbol in word):
        raise ValueError("word symbols must be in 0..alphabet_size-1")
    transitions = _transition_map(dfa)
    state = dfa.initial_state
    for symbol in word:
        state = transitions[(state, symbol)]
    return (state in dfa.accepting_states, state)


def count_accepted_words(dfa: DFA, word_length: int) -> int:
    """Count accepted words of exact length via exact integer matrix powering."""

    state_count = dfa.state_count
    if type(word_length) is not int or not 0 <= word_length <= MAX_COUNT_WORD_LENGTH:
        raise OperationDomainValidationError(
            location=("word_length",),
            code="regular_language.word_length_out_of_bounds",
            message="word length is outside the exact counting bound",
        )
    if not dfa.accepting_states:
        # Empty accepting sets accept no words of any length, including ε.
        return 0
    if word_length == 0:
        return 1 if dfa.initial_state in dfa.accepting_states else 0
    matrix_work = state_count**3 * max(1, word_length.bit_length())
    if matrix_work > MAX_COUNT_MATRIX_WORK:
        raise OperationDomainValidationError(
            location=("dfa", "word_length"),
            code="regular_language.count_work_bound",
            message="DFA matrix powering exceeds the exact work bound",
        )
    count_bits = 1 + word_length * (dfa.alphabet_size - 1).bit_length()
    count_digits = max(1, (count_bits * 30_103 + 99_999) // 100_000)
    if count_digits > MAX_COUNT_RESULT_DIGITS:
        raise OperationDomainValidationError(
            location=("dfa", "word_length"),
            code="regular_language.count_result_bound",
            message="accepted-word count exceeds the canonical result digit bound",
        )
    matrix = [[0] * state_count for _ in range(state_count)]
    for (source, _symbol), target in _transition_map(dfa).items():
        matrix[source][target] += 1
    from jacobian.math.logic.languages.regular._flint import accepted_word_count

    return accepted_word_count(
        tuple(tuple(row) for row in matrix),
        dfa.initial_state,
        dfa.accepting_states,
        word_length,
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


def _transition_parikh_profile_data(
    plan: TransitionParikhAdmissionPlan,
    source_state: int,
    target_state: int,
    path_length: int,
) -> tuple[tuple[tuple[tuple[int, ...], int], ...], int]:
    """Compute canonical profile entries inside one admitted envelope."""

    transition_count = sum(len(transitions) for transitions in plan.outgoing)
    zero_vector = tuple(0 for _ in range(transition_count))
    layer: dict[tuple[int, tuple[int, ...]], int] = {(source_state, zero_vector): 1}
    for _ in range(path_length):
        if not layer:
            break
        next_layer: dict[tuple[int, tuple[int, ...]], int] = {}
        for (state, vector), multiplicity in layer.items():
            for transition in plan.outgoing[state]:
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
    if total_path_count != plan.expected_path_count:
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

    plan = admit_transition_profile(automaton, source_state, target_state, path_length)
    target_entries, total_path_count = _transition_parikh_profile_data(
        plan, source_state, target_state, path_length
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
