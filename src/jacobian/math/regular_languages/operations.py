"""Exact finite-automaton and regular-language kernels."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
from jacobian.math.regular_languages.values import (
    DFA,
    MAX_TRANSITION_PROFILE_COUNT_DIGITS,
    MAX_TRANSITION_PROFILE_ENTRIES,
    MAX_TRANSITION_PROFILE_PATH_LENGTH,
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

# The catalog path performs at most three bounded state preflights (request,
# computation, and result binding) and two sparse profile recurrences
# (computation and source-binding replay). These per-pass limits therefore also
# give a fixed deterministic bound on the complete public call.
_MAX_TRANSITION_PROFILE_DP_UPDATES = 2_000_000
_MAX_TRANSITION_PROFILE_VECTOR_UPDATE_WORK = 20_000_000
_MAX_TRANSITION_PROFILE_VECTOR_COORDINATES = 4_000_000
_MAX_TRANSITION_PROFILE_RESULT_BYTES = 4 * 1024 * 1024
_MAX_TRANSITION_PROFILE_COMPOSITION_CELLS = 20_000_000

# Conservative canonical-wire allowances. One source transition has four
# bounded integer fields; one profile entry adds its keys, brackets, separators,
# dense coordinate vector, and exact multiplicity string. The estimates stay
# above the compact RFC 8785 encoding used at the transport boundary.
_AUTOMATON_BASE_WIRE_BYTES = 256
_AUTOMATON_TRANSITION_WIRE_BYTES = 128
_PROFILE_BASE_WIRE_BYTES = 256
_PROFILE_ENTRY_BASE_WIRE_BYTES = 96


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


def dfa_transition_carrier(dfa: DFA) -> FiniteLabeledAutomaton:
    """Project a DFA onto a labeled carrier with a stable transition axis.

    The unique DFA transitions are ordered by ``(source, symbol)`` before
    receiving contiguous transition IDs. Initial and accepting-state data are
    intentionally absent from the path carrier.
    """

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


def _path_count_and_walk_update_bound(
    automaton: FiniteLabeledAutomaton,
    source_state: int,
    target_state: int,
    path_length: int,
    composition_update_bound: int,
) -> tuple[int, int, int]:
    """Return target path count, capped path extensions, and peak layer paths."""

    outgoing = _outgoing_transitions(automaton)

    counts = {source_state: 1}
    walk_updates = 0
    max_layer_paths = 1
    for _ in range(path_length):
        next_counts: dict[int, int] = {}
        for state, multiplicity in counts.items():
            for transition in outgoing[state]:
                # Every reachable-state scan extends at least one path, so this
                # capped count also bounds the state preflight's transition work.
                walk_updates = min(
                    _MAX_TRANSITION_PROFILE_DP_UPDATES + 1,
                    walk_updates + multiplicity,
                )
                next_counts[transition.target] = (
                    next_counts.get(transition.target, 0) + multiplicity
                )
        if (
            walk_updates > _MAX_TRANSITION_PROFILE_DP_UPDATES
            and composition_update_bound > _MAX_TRANSITION_PROFILE_DP_UPDATES
        ):
            raise ValueError(
                "transition-Parikh DP transition-update bound exceeded; reduce the "
                "path length or transition branching"
            )
        counts = next_counts
        max_layer_paths = max(max_layer_paths, sum(counts.values()))
        if not counts:
            break
    return counts.get(target_state, 0), walk_updates, max_layer_paths


def _capped_combination(n: int, k: int, cap: int) -> int:
    """Return ``min(comb(n, k), cap + 1)`` without materializing huge values."""

    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for factor in range(1, k + 1):
        result = result * (n - k + factor) // factor
        if result > cap:
            return cap + 1
    return result


def _transition_profile_composition_bounds(
    transition_count: int, path_length: int
) -> tuple[int, int]:
    """Return dense-vector support and DP-update bounds from stars and bars."""

    if path_length == 0:
        return 1, 0
    if transition_count == 0:
        return 0, 0
    profile_cells = _capped_combination(
        path_length + transition_count - 1,
        transition_count - 1,
        _MAX_TRANSITION_PROFILE_COMPOSITION_CELLS,
    )
    # At layer k, each transition can extend at most C(k+m-1,m-1)
    # transition-count vectors. Summing k=0..L-1 gives m*C(L+m-1,m).
    per_transition_cap = _MAX_TRANSITION_PROFILE_DP_UPDATES // transition_count
    per_transition_updates = _capped_combination(
        path_length + transition_count - 1,
        transition_count,
        per_transition_cap,
    )
    dp_updates = min(
        _MAX_TRANSITION_PROFILE_DP_UPDATES + 1,
        transition_count * per_transition_updates,
    )
    return profile_cells, dp_updates


def _require_transition_profile_envelope(
    automaton: FiniteLabeledAutomaton,
    source_state: int,
    target_state: int,
    path_length: int,
) -> int:
    """Preflight exact work, intermediate, count-digit, and result bounds."""

    if not 0 <= source_state < automaton.state_count:
        raise ValueError("source_state must be in 0..state_count-1")
    if not 0 <= target_state < automaton.state_count:
        raise ValueError("target_state must be in 0..state_count-1")
    if path_length < 0:
        raise ValueError("path_length must be nonnegative")
    if path_length > MAX_TRANSITION_PROFILE_PATH_LENGTH:
        raise ValueError(
            "path_length exceeds the transition-Parikh preflight length bound"
        )

    transition_count = len(automaton.transitions)
    profile_composition_bound, composition_update_bound = (
        _transition_profile_composition_bounds(transition_count, path_length)
    )
    target_path_count, walk_update_bound, max_layer_paths = (
        _path_count_and_walk_update_bound(
            automaton,
            source_state,
            target_state,
            path_length,
            composition_update_bound,
        )
    )

    dp_update_bound = min(walk_update_bound, composition_update_bound)
    if dp_update_bound > _MAX_TRANSITION_PROFILE_DP_UPDATES:
        raise ValueError(
            "transition-Parikh DP transition-update bound exceeded; reduce the "
            "path length or transition branching"
        )

    profile_cell_bound = min(target_path_count, profile_composition_bound)
    if profile_cell_bound > MAX_TRANSITION_PROFILE_ENTRIES:
        raise ValueError(
            "transition-Parikh profile-cell bound exceeded; reduce the path length "
            "or transition dimension"
        )

    vector_update_work = dp_update_bound * transition_count
    if vector_update_work > _MAX_TRANSITION_PROFILE_VECTOR_UPDATE_WORK:
        raise ValueError(
            "transition-Parikh dense-vector update-work bound exceeded; reduce "
            "the transition axis or path branching"
        )

    layer_cell_bound = min(
        max_layer_paths,
        automaton.state_count * profile_composition_bound,
        dp_update_bound + 1,
    )
    vector_coordinate_bound = layer_cell_bound * transition_count
    if vector_coordinate_bound > _MAX_TRANSITION_PROFILE_VECTOR_COORDINATES:
        raise ValueError(
            "transition-Parikh intermediate vector-coordinate bound exceeded; "
            "reduce the transition axis or path branching"
        )

    count_digits = len(format_canonical_integer(max(1, target_path_count)))
    if count_digits > MAX_TRANSITION_PROFILE_COUNT_DIGITS:
        raise ValueError(
            "transition-Parikh multiplicity digit bound exceeded; reduce the path "
            "length or transition branching"
        )
    coordinate_digits = len(str(max(1, path_length)))
    estimated_entry_bytes = (
        _PROFILE_ENTRY_BASE_WIRE_BYTES
        + transition_count * (coordinate_digits + 2)
        + count_digits
    )
    estimated_result_bytes = (
        _AUTOMATON_BASE_WIRE_BYTES
        + transition_count * _AUTOMATON_TRANSITION_WIRE_BYTES
        + _PROFILE_BASE_WIRE_BYTES
        + profile_cell_bound * estimated_entry_bytes
    )
    if estimated_result_bytes > _MAX_TRANSITION_PROFILE_RESULT_BYTES:
        raise ValueError(
            "transition-Parikh serialized-result bound exceeded; reduce the "
            "transition axis or profile support"
        )
    return target_path_count


def _transition_parikh_profile_data(
    automaton: FiniteLabeledAutomaton,
    source_state: int,
    target_state: int,
    path_length: int,
) -> tuple[tuple[tuple[tuple[int, ...], int], ...], int]:
    """Compute canonical profile entries and their independent total count.

    The sparse dynamic program obeys

    ``P[k+1, target, v+e_i] += P[k, source, v]``

    for every identified transition ``i: source -> target``. Dense vectors are
    returned on the automaton's complete transition-ID axis.
    """

    expected_path_count = _require_transition_profile_envelope(
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
    return TransitionParikhProfile(
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
