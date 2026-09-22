"""Domain-owned Petri net kernels."""

from __future__ import annotations

from collections import deque

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.lattices.operations import hermite_normal_form
from jacobian.math.logic.automata.petri_nets._models import (
    MAX_FIRING_SEQUENCE_LENGTH,
    MAX_SIPHON_TRAP_PLACES,
    MAX_SIPHON_TRAP_WORK,
    EnabledTransitionsResult,
    FireTransitionResult,
    FiringSequenceReplayResult,
    IncidenceMatrixResult,
    PetriInvariantsResult,
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
from jacobian.math.matrices.certified_snf.operations import (
    smith_normal_form_certificate,
)
from jacobian.math.matrices.certified_snf.values import (
    MAX_CERTIFIED_SNF_INPUT_DIMENSION,
)
from jacobian.math.matrices.values import IntegerMatrix

__all__ = [
    "compute_incidence_matrix",
    "enabled_transitions",
    "find_minimal_siphons",
    "find_minimal_traps",
    "fire_transition",
    "petri_invariants",
    "reachability_graph",
    "replay_firing_sequence",
    "siphon_trap",
    "verify_enabled_transitions",
    "verify_fire_transition",
    "verify_firing_sequence_replay",
    "verify_incidence_matrix",
    "verify_invariants",
    "verify_reachability_graph",
    "verify_siphon_trap",
]


def _admit_net(net: object) -> PetriNet:
    if not isinstance(net, PetriNet):
        raise OperationDomainValidationError(
            location=("net",),
            code="petri_net.net_type",
            message="net must be a PetriNet value",
        )
    try:
        return PetriNet.model_validate(net.model_dump(), strict=True)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("net",),
            code="petri_net.net_shape",
            message="net must satisfy its complete canonical matrix and axis shape",
        ) from exc


def _admit_marking(marking: object, net: PetriNet) -> Marking:
    if not isinstance(marking, Marking):
        raise OperationDomainValidationError(
            location=("marking",),
            code="petri_net.marking_type",
            message="marking must be a Marking value",
        )
    try:
        admitted = Marking.model_validate(marking.model_dump(), strict=True)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("marking",),
            code="petri_net.marking_shape",
            message="marking must satisfy its complete canonical token shape",
        ) from exc
    if admitted.net is not None and admitted.net != net:
        raise OperationDomainValidationError(
            location=("marking", "net"),
            code="petri_net.marking_parent",
            message="marking belongs to a different Petri net place axis",
        )
    return admitted


def _require_marking_size(net: PetriNet, marking: object) -> Marking:
    """Require one token count for each exact net place axis."""

    admitted = _admit_marking(marking, net)
    if len(admitted.tokens) != net.place_count:
        raise OperationDomainValidationError(
            location=("marking",),
            code="petri_net.marking_axis",
            message="marking length must match place_count",
        )
    return admitted


def _bound_marking(net: PetriNet, marking: Marking, tokens: tuple[int, ...]) -> Marking:
    """Retain an explicitly supplied net parent through derived markings."""

    return Marking(tokens=tokens, net=net if marking.net is not None else None)


def _enabled_transition_indices(net: PetriNet, marking: Marking) -> list[int]:
    """Return indices of all transitions enabled at the given marking."""
    marking = _require_marking_size(net, marking)
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

    net = _admit_net(net)
    marking = _require_marking_size(net, marking)
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

    net = _admit_net(net)
    marking = _require_marking_size(net, marking)
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
        new_marking=_bound_marking(net, marking, new_tokens),
    )


def _require_sequence_axes(net: PetriNet, sequence: tuple[int, ...]) -> None:
    """Share the catalog sequence-axis admission with native callers."""

    if len(sequence) > MAX_FIRING_SEQUENCE_LENGTH:
        raise OperationResourceAdmissionError(
            location=("sequence",),
            code="petri_net.firing_sequence_length",
            message="firing sequence exceeds the admitted replay length",
        )
    for transition in sequence:
        if type(transition) is not int or not 0 <= transition < net.transition_count:
            raise OperationDomainValidationError(
                location=("sequence",),
                code="petri_net.transition_axis",
                message="sequence transitions must use the net axis",
            )


def replay_firing_sequence(
    net: PetriNet, marking: Marking, sequence: tuple[int, ...]
) -> FiringSequenceReplayResult:
    """Replay a bounded transition sequence step by step from a marking."""

    net = _admit_net(net)
    marking = _require_marking_size(net, marking)
    _require_sequence_axes(net, sequence)
    current = list(marking.tokens)
    prefix: list[Marking] = []
    parikh = [0] * net.transition_count
    for index, transition in enumerate(sequence):
        deficit = tuple(
            max(0, net.pre[place][transition] - current[place])
            for place in range(net.place_count)
        )
        if any(entry > 0 for entry in deficit):
            return FiringSequenceReplayResult._from_kernel(
                net=net,
                marking=marking,
                sequence=tuple(sequence),
                status="BLOCKED",
                prefix_markings=tuple(prefix),
                final_marking=None,
                parikh=tuple(parikh),
                state_equation_residual=tuple(
                    current[place]
                    - marking.tokens[place]
                    - sum(
                        (net.post[place][t] - net.pre[place][t]) * parikh[t]
                        for t in range(net.transition_count)
                    )
                    for place in range(net.place_count)
                ),
                blocked_index=index,
                deficit=deficit,
                first_deficient_place=next(
                    place for place, entry in enumerate(deficit) if entry > 0
                ),
            )
        current = [
            current[place] - net.pre[place][transition] + net.post[place][transition]
            for place in range(net.place_count)
        ]
        if any(token > MAX_PETRI_MARKING for token in current):
            raise OperationResourceAdmissionError(
                location=("net", "sequence"),
                code="petri_net.firing_sequence_token_bound",
                message="a replayed marking escapes the declared token envelope",
            )
        prefix.append(_bound_marking(net, marking, tuple(current)))
        parikh[transition] += 1
    final = _bound_marking(net, marking, tuple(current))
    residual = tuple(
        final.tokens[place]
        - marking.tokens[place]
        - sum(
            (net.post[place][t] - net.pre[place][t]) * parikh[t]
            for t in range(net.transition_count)
        )
        for place in range(net.place_count)
    )
    if any(entry != 0 for entry in residual):  # pragma: no cover - kernel invariant.
        raise OperationDomainValidationError(
            location=("net", "sequence"),
            code="petri_net.state_equation_mismatch",
            message="replayed firing violates the state equation",
        )
    return FiringSequenceReplayResult._from_kernel(
        net=net,
        marking=marking,
        sequence=tuple(sequence),
        status="FIRES",
        prefix_markings=tuple(prefix),
        final_marking=final,
        parikh=tuple(parikh),
        state_equation_residual=residual,
        blocked_index=None,
        deficit=None,
        first_deficient_place=None,
    )


def verify_firing_sequence_replay(claim: FiringSequenceReplayResult) -> bool:
    """Verify a firing-sequence replay against its retained source context."""

    try:
        return replay_firing_sequence(claim.net, claim.marking, claim.sequence) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def compute_incidence_matrix(net: PetriNet) -> IncidenceMatrixResult:
    """Compute C = Post - Pre."""
    net = _admit_net(net)
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
    net = _admit_net(net)
    initial_marking = _require_marking_size(net, initial_marking)
    require_reachability_bounds(net, max_states)
    initial = tuple(initial_marking.tokens)
    state_list: list[tuple[int, ...]] = [initial]
    state_index: dict[tuple[int, ...], int] = {initial: 0}
    edges: list[tuple[int, int, int]] = []
    queue: deque[int] = deque([0])
    truncated = False
    while queue:
        idx = queue.popleft()
        marking = _bound_marking(net, initial_marking, state_list[idx])
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
                marking=_bound_marking(net, initial_marking, tokens),
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
    net = _admit_net(net)
    return _minimal_place_families(net)[0]


def find_minimal_traps(net: PetriNet) -> list[frozenset[int]]:
    """Return all inclusion-minimal nonempty traps within the exact envelope."""
    net = _admit_net(net)
    return _minimal_place_families(net)[1]


def siphon_trap(net: PetriNet) -> SiphonTrapResult:
    """Return minimal families by independent place-transition components."""
    net = _admit_net(net)
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


def _admit_invariants(net: PetriNet) -> None:
    """Admit the certified Smith envelope shared with the SNF owner kernel."""

    if (
        net.place_count > MAX_CERTIFIED_SNF_INPUT_DIMENSION
        or net.transition_count > MAX_CERTIFIED_SNF_INPUT_DIMENSION
    ):
        raise OperationResourceAdmissionError(
            location=("net",),
            code="petri_net.invariants_dimension_bound",
            message=(
                "invariant computation supports at most "
                f"{MAX_CERTIFIED_SNF_INPUT_DIMENSION} places and transitions "
                "for exact certified-Smith kernel composition"
            ),
        )


def _snf_integer_kernel(rows: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Return the saturated integer basis of the kernel of ``rows``.

    Consumer composition over the certified Smith owner kernel: with
    ``D = U A V`` and rank ``r``, the final ``n - r`` columns of the
    unimodular ``V`` span ``ker_Z(A)``.
    """

    matrix = IntegerMatrix(
        row_count=len(rows),
        column_count=len(rows[0]),
        entries=tuple(tuple(row) for row in rows),
    )
    certificate = smith_normal_form_certificate(matrix)
    right = certificate.right_transformation.entries
    ambient = len(rows[0])
    return tuple(
        tuple(int(right[row][column]) for row in range(ambient))
        for column in range(certificate.rank, ambient)
    )


def _canonicalize_invariant_basis(
    vectors: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Canonicalize one invariant basis through the HNF owner kernel.

    The row Hermite normal form of the stacked basis spans the same integer
    module; rows are sign-normalized and sorted for a deterministic carrier.
    A failed HNF admission falls back to the sign-normalized SNF basis,
    which the kernel replays either way.
    """

    if not vectors:
        return ()
    try:
        normal, _transformation = hermite_normal_form(
            [list(vector) for vector in vectors]
        )
        rows = [
            [int(normal[row, column]) for column in range(normal.ncols())]
            for row in range(normal.nrows())
        ]
    except (ValueError, TypeError, ArithmeticError, OperationDomainValidationError):
        rows = [list(vector) for vector in vectors]
    canonical: list[tuple[int, ...]] = []
    for row in rows:
        if not any(row):
            continue
        if next(value for value in row if value) < 0:
            row = [-value for value in row]
        canonical.append(tuple(row))
    return tuple(sorted(set(canonical)))


def _replay_invariants(
    incidence: list[list[int]],
    p_invariants: tuple[tuple[int, ...], ...],
    t_invariants: tuple[tuple[int, ...], ...],
) -> None:
    """Replay every invariant against the incidence matrix in the kernel."""

    places = len(incidence)
    transitions = len(incidence[0]) if incidence else 0
    for vector in t_invariants:
        if any(
            sum(incidence[place][t] * vector[t] for t in range(transitions))
            for place in range(places)
        ):
            raise RuntimeError("a T-invariant escapes the incidence kernel")
    for vector in p_invariants:
        if any(
            sum(vector[place] * incidence[place][t] for place in range(places))
            for t in range(transitions)
        ):
            raise RuntimeError("a P-invariant escapes the left incidence kernel")


def petri_invariants(net: PetriNet) -> PetriInvariantsResult:
    """Compute P-invariants and T-invariants as exact integer modules.

    P-invariants span ``ker_Z(C^T)`` and T-invariants span ``ker_Z(C)``
    for the incidence matrix ``C``. Kernel bases come from the certified
    Smith owner kernel, canonicalized through the Hermite owner kernel;
    every returned vector is replayed against ``C`` inside this kernel.
    """

    net = _admit_net(net)
    incidence_matrix = compute_incidence_matrix(net).incidence
    incidence = [[int(value) for value in row] for row in incidence_matrix.entries]
    places = net.place_count
    transitions = net.transition_count
    if places == 0 or transitions == 0:
        p_basis = (
            ()
            if places == 0
            else tuple(
                sorted(
                    tuple(1 if place == index else 0 for place in range(places))
                    for index in range(places)
                )
            )
        )
        t_basis = (
            ()
            if transitions == 0
            else tuple(
                sorted(
                    tuple(1 if t == index else 0 for t in range(transitions))
                    for index in range(transitions)
                )
            )
        )
        rank = 0
    else:
        _admit_invariants(net)
        t_basis = _canonicalize_invariant_basis(_snf_integer_kernel(incidence))
        transposed = [
            [incidence[place][t] for place in range(places)] for t in range(transitions)
        ]
        p_basis = _canonicalize_invariant_basis(_snf_integer_kernel(transposed))
        # Both certificates see the same rank r: len(t) = t - r and
        # len(p) = p - r, so r = min(p, t) - min(len(p), len(t)). The shape
        # check below rejects any disagreement between the two owners.
        rank = min(places, transitions) - min(len(p_basis), len(t_basis))
        if len(t_basis) != transitions - rank or len(p_basis) != places - rank:
            raise RuntimeError("invariant bases disagree with the Smith rank")
    _replay_invariants(incidence, p_basis, t_basis)
    return PetriInvariantsResult._from_kernel(
        net=net,
        incidence=incidence_matrix,
        incidence_rank=rank,
        p_invariants=p_basis,
        t_invariants=t_basis,
        replayed=True,
    )


def verify_invariants(claim: PetriInvariantsResult) -> bool:
    """Verify P/T-invariant bases against their retained net."""

    try:
        return petri_invariants(claim.net) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
