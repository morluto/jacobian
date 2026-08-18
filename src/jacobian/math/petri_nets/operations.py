"""Domain-owned Petri net kernels."""

from __future__ import annotations

from collections import deque

from jacobian.math.petri_nets.values import Marking, PetriNet

__all__ = [
    "compute_incidence_matrix",
    "enabled_transitions",
    "fire_transition",
    "reachability_graph",
]


def enabled_transitions(net: PetriNet, marking: Marking) -> list[int]:
    """Return indices of all transitions enabled at the given marking."""
    _require_marking_size(net, marking)
    result: list[int] = []
    for t in range(net.transition_count):
        enabled = True
        for p in range(net.place_count):
            if marking.tokens[p] < net.pre[p][t]:
                enabled = False
                break
        if enabled:
            result.append(t)
    return result


def fire_transition(
    net: PetriNet,
    marking: Marking,
    transition: int,
) -> tuple[bool, tuple[int, ...]]:
    """Fire a transition. Returns (success, new_marking)."""
    _require_marking_size(net, marking)
    if not 0 <= transition < net.transition_count:
        raise ValueError("transition index is out of range")
    for p in range(net.place_count):
        if marking.tokens[p] < net.pre[p][transition]:
            return (False, marking.tokens)
    new_tokens = tuple(
        marking.tokens[p] - net.pre[p][transition] + net.post[p][transition]
        for p in range(net.place_count)
    )
    return (True, new_tokens)


def compute_incidence_matrix(net: PetriNet) -> tuple[tuple[int, ...], ...]:
    """Compute C = Post - Pre."""
    return tuple(
        tuple(net.post[p][t] - net.pre[p][t] for t in range(net.transition_count))
        for p in range(net.place_count)
    )


def reachability_graph(
    net: PetriNet,
    initial_marking: Marking,
    max_states: int = 10000,
) -> tuple[
    list[tuple[int, ...]],
    list[tuple[int, int, int]],
    list[tuple[int, int, tuple[int, ...]]],
]:
    """Compute the bounded reachability graph via BFS.

    Returns (states, edges, frontier).
    Each edge is (source_index, transition, target_index).
    Each frontier record is an enabled firing whose target could not be
    admitted because ``max_states`` was exhausted.
    """
    _require_marking_size(net, initial_marking)
    if not 1 <= max_states <= 100000:
        raise ValueError("max_states must be between 1 and 100000")
    initial = tuple(initial_marking.tokens)
    state_list: list[tuple[int, ...]] = [initial]
    state_index: dict[tuple[int, ...], int] = {initial: 0}
    edges: list[tuple[int, int, int]] = []
    frontier: list[tuple[int, int, tuple[int, ...]]] = []
    queue: deque[int] = deque([0])
    while queue:
        idx = queue.popleft()
        marking = Marking(tokens=state_list[idx])
        enabled = enabled_transitions(net, marking)
        for t in enabled:
            success, new_tokens = fire_transition(net, marking, t)
            if not success:
                continue
            if new_tokens not in state_index:
                if len(state_list) >= max_states:
                    frontier.append((idx, t, new_tokens))
                    continue
                state_index[new_tokens] = len(state_list)
                state_list.append(new_tokens)
                queue.append(len(state_list) - 1)
            edges.append((idx, t, state_index[new_tokens]))
    return (state_list, edges, frontier)


def _require_marking_size(net: PetriNet, marking: Marking) -> None:
    if len(marking.tokens) != net.place_count:
        raise ValueError("marking length must match place_count")
