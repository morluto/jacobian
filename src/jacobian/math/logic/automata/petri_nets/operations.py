"""Domain-owned Petri net kernels."""

from __future__ import annotations

from collections import deque

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.automata.petri_nets._models import (
    MAX_SIPHON_TRAP_PLACES,
    MAX_SIPHON_TRAP_WORK,
    EnabledTransitionsResult,
    FireTransitionResult,
    IncidenceMatrixResult,
    PetriMarkingState,
    PetriPlaceSubset,
    PetriReachabilityEdge,
    ReachabilityResult,
    SiphonTrapResult,
)
from jacobian.math.logic.automata.petri_nets.values import (
    MAX_PETRI_MARKING,
    Marking,
    PetriNet,
    require_reachability_bounds,
)
from jacobian.math.matrices.values import IntegerMatrix

__all__ = [
    "compute_incidence_matrix",
    "enabled_transitions",
    "find_minimal_siphons",
    "find_minimal_traps",
    "fire_transition",
    "reachability_graph",
    "siphon_trap",
    "verify_enabled_transitions",
    "verify_fire_transition",
    "verify_incidence_matrix",
    "verify_reachability_graph",
    "verify_siphon_trap",
]


def _require_marking_size(net: PetriNet, marking: Marking) -> None:
    """Require one token count for each place in the net."""

    if len(marking.tokens) != net.place_count:
        raise OperationDomainValidationError(
            location=("marking",),
            code="petri_net.marking_axis",
            message="marking length must match place_count",
        )


def _enabled_transition_indices(net: PetriNet, marking: Marking) -> list[int]:
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


def enabled_transitions(net: PetriNet, marking: Marking) -> EnabledTransitionsResult:
    """Return all transitions enabled at the given marking."""

    return EnabledTransitionsResult(
        net=net,
        marking=marking,
        transitions=tuple(_enabled_transition_indices(net, marking)),
    )


def _fire_transition_tokens(
    net: PetriNet,
    marking: Marking,
    transition: int,
) -> tuple[bool, tuple[int, ...]]:
    """Return whether a transition fired and its successor token tuple."""

    _require_marking_size(net, marking)
    if not 0 <= transition < net.transition_count:
        raise OperationDomainValidationError(
            location=("transition",),
            code="petri_net.transition_axis",
            message="transition index out of range",
        )
    for p in range(net.place_count):
        if marking.tokens[p] < net.pre[p][transition]:
            return (False, marking.tokens)
    new_tokens = tuple(
        marking.tokens[p] - net.pre[p][transition] + net.post[p][transition]
        for p in range(net.place_count)
    )
    return (True, new_tokens)


def fire_transition(
    net: PetriNet,
    marking: Marking,
    transition: int,
) -> FireTransitionResult:
    """Fire one transition and return its canonical bounded outcome."""

    success, new_tokens = _fire_transition_tokens(net, marking, transition)
    if any(token > MAX_PETRI_MARKING for token in new_tokens):
        return FireTransitionResult(
            net=net,
            marking=marking,
            transition=transition,
            status="ESCAPES_DECLARED_ENVELOPE",
            envelope_escape=new_tokens,
        )
    return FireTransitionResult(
        net=net,
        marking=marking,
        transition=transition,
        status="FIRED" if success else "NOT_ENABLED",
        new_marking=Marking(tokens=new_tokens),
    )


def compute_incidence_matrix(net: PetriNet) -> IncidenceMatrixResult:
    """Compute C = Post - Pre."""
    return IncidenceMatrixResult(
        net=net,
        incidence=IntegerMatrix(
            row_count=net.place_count,
            column_count=net.transition_count,
            entries=tuple(
                tuple(
                    net.post[p][t] - net.pre[p][t] for t in range(net.transition_count)
                )
                for p in range(net.place_count)
            ),
        ),
    )


def reachability_graph(
    net: PetriNet,
    initial_marking: Marking,
    max_states: int = 10000,
) -> ReachabilityResult:
    """Compute the bounded reachability graph via BFS.

    Returns (states, edges, truncated).
    Each edge is (source_index, transition, target_index).
    """
    _require_marking_size(net, initial_marking)
    require_reachability_bounds(net, max_states)
    initial = tuple(initial_marking.tokens)
    state_list: list[tuple[int, ...]] = [initial]
    state_index: dict[tuple[int, ...], int] = {initial: 0}
    edges: list[tuple[int, int, int]] = []
    queue: deque[int] = deque([0])
    truncated = False
    while queue:
        idx = queue.popleft()
        marking = Marking(tokens=state_list[idx])
        enabled = _enabled_transition_indices(net, marking)
        for t in enabled:
            success, new_tokens = _fire_transition_tokens(net, marking, t)
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
    return ReachabilityResult(
        net=net,
        initial_marking=initial_marking,
        max_states=max_states,
        states=tuple(
            PetriMarkingState(
                state_index=index,
                place_axis=tuple(range(net.place_count)),
                marking=Marking(tokens=tokens),
            )
            for index, tokens in enumerate(state_list)
        ),
        edges=tuple(
            PetriReachabilityEdge(
                source_state=source, transition=transition, target_state=target
            )
            for source, transition, target in edges
        ),
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# Siphon and trap detection
# ---------------------------------------------------------------------------


def _pre_places(net: PetriNet, t: int) -> frozenset[int]:
    """Return places that have an input arc to transition t."""
    return frozenset(p for p in range(net.place_count) if net.pre[p][t] > 0)


def _post_places(net: PetriNet, t: int) -> frozenset[int]:
    """Return places that have an output arc from transition t."""
    return frozenset(p for p in range(net.place_count) if net.post[p][t] > 0)


def _place_components(
    net: PetriNet,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Partition places joined by the support of any transition."""
    neighbors: list[set[int]] = [set() for _ in range(net.place_count)]
    supports = [
        tuple(sorted(_pre_places(net, t) | _post_places(net, t)))
        for t in range(net.transition_count)
    ]
    for support in supports:
        if support:
            anchor = support[0]
            for place in support[1:]:
                neighbors[anchor].add(place)
                neighbors[place].add(anchor)
    unseen = set(range(net.place_count))
    components = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        places = {start}
        frontier = [start]
        while frontier:
            for neighbor in neighbors[frontier.pop()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    places.add(neighbor)
                    frontier.append(neighbor)
        transitions = tuple(
            t for t, support in enumerate(supports) if support and support[0] in places
        )
        components.append((tuple(sorted(places)), transitions))
    return tuple(components)


def _minimal_component_families(
    net: PetriNet, places: tuple[int, ...], transitions: tuple[int, ...]
) -> tuple[list[frozenset[int]], list[frozenset[int]]]:
    """Enumerate local subsets with a subset-DP minimality certificate."""
    if not transitions:
        singletons = [frozenset((place,)) for place in places]
        return singletons, singletons.copy()
    arcs = tuple(
        (
            sum(1 << i for i, p in enumerate(places) if net.pre[p][t]),
            sum(1 << i for i, p in enumerate(places) if net.post[p][t]),
        )
        for t in transitions
    )
    # Bit1/bit2 record whether any nonempty subset is a siphon/trap.
    # All proper submasks precede mask. Their union supplies minimality
    # in O(number of places), avoiding pairwise scans over found families.
    contains = bytearray(1 << len(places))
    siphons: list[frozenset[int]] = []
    traps: list[frozenset[int]] = []
    for mask in range(1, len(contains)):
        inherited = 0
        bits = mask
        while bits:
            bit = bits & -bits
            inherited |= contains[mask ^ bit]
            bits ^= bit
        valid = 3
        for pre, post in arcs:
            if mask & post and not mask & pre:
                valid &= ~1
            if mask & pre and not mask & post:
                valid &= ~2
        minimal = valid & ~inherited
        if minimal:
            subset = frozenset(p for i, p in enumerate(places) if mask & (1 << i))
            if minimal & 1:
                siphons.append(subset)
            if minimal & 2:
                traps.append(subset)
        contains[mask] = inherited | valid
    return siphons, traps


def _minimal_place_families(
    net: PetriNet,
) -> tuple[list[frozenset[int]], list[frozenset[int]]]:
    """Admit independent component searches before allocating subset tables."""
    components = _place_components(net)
    work = 2 * net.place_count * net.transition_count
    for places, transitions in components:
        if not transitions:
            work += 2 * len(places)
            continue
        if len(places) > MAX_SIPHON_TRAP_PLACES:
            raise OperationResourceAdmissionError(
                location=("net",),
                code="petri_net.siphon_trap_place_bound",
                message=f"each coupled siphon/trap component supports at most {MAX_SIPHON_TRAP_PLACES} places for exact enumeration",
            )
        candidates = (1 << len(places)) - 1
        work += 2 * candidates * (len(transitions) + len(places))
    if work > MAX_SIPHON_TRAP_WORK:
        raise OperationResourceAdmissionError(
            location=("net",),
            code="petri_net.siphon_trap_work_bound",
            message="siphon/trap component subset work exceeds the admitted bound",
        )
    siphons: list[frozenset[int]] = []
    traps: list[frozenset[int]] = []
    for places, transitions in components:
        local_siphons, local_traps = _minimal_component_families(
            net, places, transitions
        )
        siphons.extend(local_siphons)
        traps.extend(local_traps)

    # A global minimal family member lies in one component: each nonempty
    # component intersection separately satisfies the same implications.
    def key(subset: frozenset[int]) -> tuple[int, tuple[int, ...]]:
        return len(subset), tuple(sorted(subset))

    return sorted(siphons, key=key), sorted(traps, key=key)


def find_minimal_siphons(net: PetriNet) -> list[frozenset[int]]:
    """Return all inclusion-minimal nonempty siphons within the exact envelope."""
    return _minimal_place_families(net)[0]


def find_minimal_traps(net: PetriNet) -> list[frozenset[int]]:
    """Return all inclusion-minimal nonempty traps within the exact envelope."""
    return _minimal_place_families(net)[1]


def siphon_trap(net: PetriNet) -> SiphonTrapResult:
    """Return minimal families by independent place-transition components."""
    siphons, traps = _minimal_place_families(net)
    return SiphonTrapResult(
        net=net,
        siphons=tuple(PetriPlaceSubset(places=tuple(sorted(s))) for s in siphons),
        traps=tuple(PetriPlaceSubset(places=tuple(sorted(t))) for t in traps),
    )


def verify_enabled_transitions(claim: EnabledTransitionsResult) -> bool:
    """Verify enabledness against the retained net and marking."""

    try:
        return enabled_transitions(claim.net, claim.marking) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_fire_transition(claim: FireTransitionResult) -> bool:
    """Verify a firing outcome against its retained source context."""

    try:
        return fire_transition(claim.net, claim.marking, claim.transition) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_incidence_matrix(claim: IncidenceMatrixResult) -> bool:
    """Verify an incidence matrix against its retained net."""

    try:
        return compute_incidence_matrix(claim.net) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_reachability_graph(claim: ReachabilityResult) -> bool:
    """Verify a bounded reachability graph against its retained experiment."""

    try:
        return (
            reachability_graph(claim.net, claim.initial_marking, claim.max_states)
            == claim
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_siphon_trap(claim: SiphonTrapResult) -> bool:
    """Verify siphon and trap families against the retained net."""

    try:
        return siphon_trap(claim.net) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
