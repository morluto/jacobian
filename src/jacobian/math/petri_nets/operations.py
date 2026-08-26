"""Domain-owned Petri net kernels."""

from __future__ import annotations

from collections import deque

from jacobian.math.petri_nets.values import MAX_PETRI_MARKING, Marking, PetriNet

__all__ = [
    "compute_incidence_matrix",
    "enabled_transitions",
    "find_minimal_siphons",
    "find_minimal_traps",
    "fire_transition",
    "reachability_graph",
]


def enabled_transitions(net: PetriNet, marking: Marking) -> list[int]:
    """Return indices of all transitions enabled at the given marking."""
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
    if not 0 <= transition < net.transition_count:
        raise ValueError("transition index out of range")
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
) -> tuple[list[tuple[int, ...]], list[tuple[int, int, int]], bool]:
    """Compute the bounded reachability graph via BFS.

    Returns (states, edges, truncated).
    Each edge is (source_index, transition, target_index).
    """
    initial = tuple(initial_marking.tokens)
    state_list: list[tuple[int, ...]] = [initial]
    state_index: dict[tuple[int, ...], int] = {initial: 0}
    edges: list[tuple[int, int, int]] = []
    queue: deque[int] = deque([0])
    truncated = False
    while queue:
        idx = queue.popleft()
        marking = Marking(tokens=state_list[idx])
        enabled = enabled_transitions(net, marking)
        for t in enabled:
            success, new_tokens = fire_transition(net, marking, t)
            if not success:
                continue
            if any(token > MAX_PETRI_MARKING for token in new_tokens):
                truncated = True
                continue
            if new_tokens not in state_index:
                if len(state_list) >= max_states:
                    truncated = True
                    continue
                state_index[new_tokens] = len(state_list)
                state_list.append(new_tokens)
                queue.append(len(state_list) - 1)
            edges.append((idx, t, state_index[new_tokens]))
    return (state_list, edges, truncated)


# ---------------------------------------------------------------------------
# Siphon and trap detection
# ---------------------------------------------------------------------------


def _pre_places(net: PetriNet, t: int) -> frozenset[int]:
    """Return places that have an input arc to transition t."""
    return frozenset(p for p in range(net.place_count) if net.pre[p][t] > 0)


def _post_places(net: PetriNet, t: int) -> frozenset[int]:
    """Return places that have an output arc from transition t."""
    return frozenset(p for p in range(net.place_count) if net.post[p][t] > 0)


def find_minimal_siphons(net: PetriNet) -> list[frozenset[int]]:
    """Find all minimal siphons of a Petri net.

    A siphon is a non-empty set of places S such that every transition
    that outputs to S also inputs from S.  Formally, for every transition
    *t*: if ``post(t) ∩ S ≠ ∅`` then ``pre(t) ∩ S ≠ ∅``.

    Once a siphon loses all its tokens no transition can ever produce a
    token in it again.

    Returns the list of inclusion-minimal siphons.
    """
    from itertools import combinations

    n = net.place_count
    if n == 0:
        return []

    pre = [_pre_places(net, t) for t in range(net.transition_count)]
    post = [_post_places(net, t) for t in range(net.transition_count)]

    found: list[frozenset[int]] = []

    for size in range(1, n + 1):
        for subset in combinations(range(n), size):
            s = frozenset(subset)
            is_siphon = True
            for t in range(net.transition_count):
                if s & post[t] and not (s & pre[t]):
                    is_siphon = False
                    break
            if not is_siphon:
                continue
            if any(prev <= s for prev in found):
                continue
            found.append(s)

    return found


def find_minimal_traps(net: PetriNet) -> list[frozenset[int]]:
    """Find all minimal traps of a Petri net.

    A trap is a non-empty set of places S such that every transition
    that inputs from S also outputs to S.  Formally, for every transition
    *t*: if ``pre(t) ∩ S ≠ ∅`` then ``post(t) ∩ S ≠ ∅``.

    Once a trap has a token it can never become empty.

    Returns the list of inclusion-minimal traps.
    """
    from itertools import combinations

    n = net.place_count
    if n == 0:
        return []

    pre = [_pre_places(net, t) for t in range(net.transition_count)]
    post = [_post_places(net, t) for t in range(net.transition_count)]

    found: list[frozenset[int]] = []

    for size in range(1, n + 1):
        for subset in combinations(range(n), size):
            s = frozenset(subset)
            is_trap = True
            for t in range(net.transition_count):
                if s & pre[t] and not (s & post[t]):
                    is_trap = False
                    break
            if not is_trap:
                continue
            if any(prev <= s for prev in found):
                continue
            found.append(s)

    return found
