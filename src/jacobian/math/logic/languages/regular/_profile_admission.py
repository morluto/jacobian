"""Request-scoped bounds and prepared state for transition-Parikh profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.languages.regular.values import (
    MAX_TRANSITION_PROFILE_COUNT_DIGITS,
    MAX_TRANSITION_PROFILE_ENTRIES,
    MAX_TRANSITION_PROFILE_PATH_LENGTH,
    AutomatonTransition,
    FiniteLabeledAutomaton,
)


def _reject(message: str, *, resource: bool = True) -> NoReturn:
    error_type = (
        OperationResourceAdmissionError if resource else OperationDomainValidationError
    )
    raise error_type(
        location=("automaton", "source_state", "target_state", "path_length"),
        code="regular_language.transition_profile_not_admitted",
        message=message,
    )


_MAX_DP_UPDATES = 2_000_000
_MAX_VECTOR_UPDATE_WORK = 20_000_000
_MAX_VECTOR_COORDINATES = 4_000_000
_MAX_COMPOSITION_CELLS = 20_000_000


@dataclass(frozen=True)
class TransitionParikhAdmissionPlan:
    """One admitted transition-profile execution envelope."""

    expected_path_count: int
    transition_count: int
    outgoing: tuple[tuple[AutomatonTransition, ...], ...]


def _outgoing(
    automaton: FiniteLabeledAutomaton,
) -> tuple[tuple[AutomatonTransition, ...], ...]:
    result: list[list[AutomatonTransition]] = [[] for _ in range(automaton.state_count)]
    for transition in automaton.transitions:
        result[transition.source].append(transition)
    return tuple(tuple(transitions) for transitions in result)


def _relevant_outgoing(
    automaton: FiniteLabeledAutomaton, source: int, target: int
) -> tuple[tuple[AutomatonTransition, ...], ...]:
    outgoing = _outgoing(automaton)
    incoming: list[list[int]] = [[] for _ in outgoing]
    for transition in automaton.transitions:
        incoming[transition.target].append(transition.source)
    reaches_target = {target}
    pending = [target]
    while pending:
        for state in incoming[pending.pop()]:
            if state not in reaches_target:
                reaches_target.add(state)
                pending.append(state)
    reachable = {source} if source in reaches_target else set()
    pending = list(reachable)
    while pending:
        for transition in outgoing[pending.pop()]:
            state = transition.target
            if state in reaches_target and state not in reachable:
                reachable.add(state)
                pending.append(state)
    return tuple(
        tuple(t for t in row if t.target in reachable) if state in reachable else ()
        for state, row in enumerate(outgoing)
    )


def _forced_state_bounds(
    outgoing: tuple[tuple[AutomatonTransition, ...], ...], source: int, length: int
) -> tuple[int, int] | None:
    # When each state's arrows share a target, the state sequence is forced.
    # With v visits and k outgoing arrows, its count coordinates have at most
    # C(v+k-1,k-1) compositions. Multiply these independent upper bounds
    # across states, and charge each layer's actual outgoing multiplicity.
    if any(len({t.target for t in row}) > 1 for row in outgoing):
        return None
    visits = [0] * len(outgoing)
    cells = maximum = 1
    updates = 0
    state = source
    for _ in range(length):
        row = outgoing[state]
        if not row:
            break
        updates = min(_MAX_DP_UPDATES + 1, updates + cells * len(row))
        cells = cells * (visits[state] + len(row)) // (visits[state] + 1)
        visits[state] += 1
        maximum = max(maximum, cells)
        if maximum > _MAX_COMPOSITION_CELLS:
            return _MAX_COMPOSITION_CELLS + 1, _MAX_DP_UPDATES + 1
        state = row[0].target
    return maximum, updates


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
            _reject(
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
        _reject("source_state must be in 0..state_count-1", resource=False)
    if not 0 <= target_state < automaton.state_count:
        _reject("target_state must be in 0..state_count-1", resource=False)
    if path_length < 0:
        _reject("path_length must be nonnegative", resource=False)
    if path_length > MAX_TRANSITION_PROFILE_PATH_LENGTH:
        _reject("path_length exceeds the transition-Parikh preflight length bound")
    transition_count = len(automaton.transitions)
    outgoing = _relevant_outgoing(automaton, source_state, target_state)
    active_transition_count = sum(map(len, outgoing))
    composition_cells, composition_updates = _composition_bounds(
        active_transition_count, path_length
    )
    forced = _forced_state_bounds(outgoing, source_state, path_length)
    if forced is not None:
        composition_cells = min(composition_cells, forced[0])
        composition_updates = min(composition_updates, forced[1])
    target_count, walk_updates, max_layer_paths = _path_count_and_walk_bound(
        outgoing, source_state, target_state, path_length, composition_updates
    )
    updates = min(walk_updates, composition_updates)
    if updates > _MAX_DP_UPDATES:
        _reject(
            "transition-Parikh DP transition-update bound exceeded; reduce the "
            "path length or transition branching"
        )
    profile_cells = min(target_count, composition_cells)
    if profile_cells > MAX_TRANSITION_PROFILE_ENTRIES:
        _reject(
            "transition-Parikh profile-cell bound exceeded; reduce the path length "
            "or transition dimension"
        )
    if updates * transition_count > _MAX_VECTOR_UPDATE_WORK:
        _reject(
            "transition-Parikh dense-vector update-work bound exceeded; reduce "
            "the transition axis or path branching"
        )
    layer_cells = min(
        max_layer_paths,
        automaton.state_count * composition_cells,
        updates + 1,
    )
    if layer_cells * transition_count > _MAX_VECTOR_COORDINATES:
        _reject(
            "transition-Parikh intermediate vector-coordinate bound exceeded; "
            "reduce the transition axis or path branching"
        )
    count_digits = len(format_canonical_integer(max(1, target_count)))
    if count_digits > MAX_TRANSITION_PROFILE_COUNT_DIGITS:
        _reject(
            "transition-Parikh multiplicity digit bound exceeded; reduce the path "
            "length or transition branching"
        )
    return TransitionParikhAdmissionPlan(
        expected_path_count=target_count,
        transition_count=transition_count,
        outgoing=outgoing,
    )


__all__ = [
    "TransitionParikhAdmissionPlan",
    "admit_transition_profile",
]
