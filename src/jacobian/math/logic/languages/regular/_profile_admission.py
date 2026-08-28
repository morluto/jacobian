"""Bounded admission for transition-Parikh profiles.

This is intentionally separate from ``_admission``: catalog registration
imports owner tools, while the operation owner constructs the per-call plan.
"""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.canonical import format_canonical_integer
from jacobian.math.logic.languages.regular.values import (
    MAX_TRANSITION_PROFILE_COUNT_DIGITS,
    MAX_TRANSITION_PROFILE_ENTRIES,
    MAX_TRANSITION_PROFILE_PATH_LENGTH,
    AutomatonTransition,
    FiniteLabeledAutomaton,
)

_MAX_DP_UPDATES = 2_000_000
_MAX_VECTOR_UPDATE_WORK = 20_000_000
_MAX_VECTOR_COORDINATES = 4_000_000
MAX_TRANSITION_PROFILE_RESULT_BYTES = 4 * 1024 * 1024
_MAX_COMPOSITION_CELLS = 20_000_000
_AUTOMATON_BASE_WIRE_BYTES = 256
_AUTOMATON_TRANSITION_WIRE_BYTES = 128
_PROFILE_BASE_WIRE_BYTES = 256
_PROFILE_ENTRY_BASE_WIRE_BYTES = 96


@dataclass(frozen=True)
class TransitionParikhAdmissionPlan:
    """One admitted transition-profile execution envelope."""

    expected_path_count: int
    outgoing: tuple[tuple[AutomatonTransition, ...], ...]


def _outgoing(
    automaton: FiniteLabeledAutomaton,
) -> tuple[tuple[AutomatonTransition, ...], ...]:
    result: list[list[AutomatonTransition]] = [[] for _ in range(automaton.state_count)]
    for transition in automaton.transitions:
        result[transition.source].append(transition)
    return tuple(tuple(transitions) for transitions in result)


def _capped_combination(n: int, k: int, cap: int) -> int:
    if k < 0 or k > n:
        return 0
    result = 1
    for factor in range(1, min(k, n - k) + 1):
        result = result * (n - min(k, n - k) + factor) // factor
        if result > cap:
            return cap + 1
    return result


def _composition_bounds(transition_count: int, path_length: int) -> tuple[int, int]:
    if path_length == 0:
        return 1, 0
    if transition_count == 0:
        return 0, 0
    cells = _capped_combination(
        path_length + transition_count - 1,
        transition_count - 1,
        _MAX_COMPOSITION_CELLS,
    )
    per_transition = _capped_combination(
        path_length + transition_count - 1,
        transition_count,
        _MAX_DP_UPDATES // transition_count,
    )
    return cells, min(_MAX_DP_UPDATES + 1, transition_count * per_transition)


def _path_count_and_walk_bound(
    outgoing: tuple[tuple[AutomatonTransition, ...], ...],
    source_state: int,
    target_state: int,
    path_length: int,
    composition_updates: int,
) -> tuple[int, int, int]:
    counts = {source_state: 1}
    updates = 0
    max_layer_paths = 1
    for _ in range(path_length):
        next_counts: dict[int, int] = {}
        for state, multiplicity in counts.items():
            for transition in outgoing[state]:
                updates = min(_MAX_DP_UPDATES + 1, updates + multiplicity)
                next_counts[transition.target] = (
                    next_counts.get(transition.target, 0) + multiplicity
                )
        if updates > _MAX_DP_UPDATES and composition_updates > _MAX_DP_UPDATES:
            raise ValueError(
                "transition-Parikh DP transition-update bound exceeded; reduce the "
                "path length or transition branching"
            )
        counts = next_counts
        max_layer_paths = max(max_layer_paths, sum(counts.values()))
        if not counts:
            break
    return counts.get(target_state, 0), updates, max_layer_paths


def admit_transition_profile(
    automaton: FiniteLabeledAutomaton,
    source_state: int,
    target_state: int,
    path_length: int,
) -> TransitionParikhAdmissionPlan:
    """Admit one exact transition-profile computation and retain its plan."""

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
    outgoing = _outgoing(automaton)
    composition_cells, composition_updates = _composition_bounds(
        transition_count, path_length
    )
    target_count, walk_updates, max_layer_paths = _path_count_and_walk_bound(
        outgoing, source_state, target_state, path_length, composition_updates
    )
    updates = min(walk_updates, composition_updates)
    if updates > _MAX_DP_UPDATES:
        raise ValueError(
            "transition-Parikh DP transition-update bound exceeded; reduce the "
            "path length or transition branching"
        )
    profile_cells = min(target_count, composition_cells)
    if profile_cells > MAX_TRANSITION_PROFILE_ENTRIES:
        raise ValueError(
            "transition-Parikh profile-cell bound exceeded; reduce the path length "
            "or transition dimension"
        )
    if updates * transition_count > _MAX_VECTOR_UPDATE_WORK:
        raise ValueError(
            "transition-Parikh dense-vector update-work bound exceeded; reduce "
            "the transition axis or path branching"
        )
    layer_cells = min(
        max_layer_paths,
        automaton.state_count * composition_cells,
        updates + 1,
    )
    if layer_cells * transition_count > _MAX_VECTOR_COORDINATES:
        raise ValueError(
            "transition-Parikh intermediate vector-coordinate bound exceeded; "
            "reduce the transition axis or path branching"
        )
    count_digits = len(format_canonical_integer(max(1, target_count)))
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
    estimated_bytes = (
        _AUTOMATON_BASE_WIRE_BYTES
        + transition_count * _AUTOMATON_TRANSITION_WIRE_BYTES
        + _PROFILE_BASE_WIRE_BYTES
        + profile_cells * estimated_entry_bytes
    )
    if estimated_bytes > MAX_TRANSITION_PROFILE_RESULT_BYTES:
        raise ValueError(
            "transition-Parikh serialized-result bound exceeded; reduce the "
            "transition axis or profile support"
        )
    return TransitionParikhAdmissionPlan(
        expected_path_count=target_count,
        outgoing=outgoing,
    )


__all__ = [
    "MAX_TRANSITION_PROFILE_RESULT_BYTES",
    "TransitionParikhAdmissionPlan",
    "admit_transition_profile",
]
