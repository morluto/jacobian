"""Domain-owned Petri net kernels."""

from __future__ import annotations

from collections import deque
from itertools import combinations
from math import comb
from typing import Literal

from jacobian.canonical import strict_json_object_size
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.lattices.operations import hermite_normal_form
from jacobian.math.logic.automata.petri_nets._models import (
    MAX_CONCURRENT_STEP_OCCURRENCES,
    MAX_FIRING_SEQUENCE_LENGTH,
    MAX_MARKING_COMMUTATION_PROFILE_OUTPUT_BYTES,
    MAX_MARKING_COMMUTATION_PROFILE_WORK,
    MAX_MARKING_CONFLICT_PROFILE_OUTPUT_BYTES,
    MAX_PETRI_NONNEGATIVE_INVARIANT_OUTPUT_BYTES,
    MAX_PETRI_NONNEGATIVE_INVARIANT_WORK,
    MAX_SIPHON_TRAP_FAMILY_OUTPUT_BYTES,
    MAX_SIPHON_TRAP_PLACES,
    MAX_SIPHON_TRAP_WORK,
    MAX_STATE_EQUATION_OCCURRENCES,
    ConcurrentStepResult,
    EnabledTransitionsResult,
    FireTransitionResult,
    FiringSequenceReplayResult,
    IncidenceMatrixResult,
    MarkingCommutationProfileResult,
    MarkingConflictProfileResult,
    MarkingReachabilityResult,
    PetriInvariantsResult,
    PetriMarkingState,
    PetriNonnegativeInvariantResult,
    PetriPlaceSubset,
    PetriReachabilityEdge,
    PlaceSetInitialMarkingProfileResult,
    PlaceSetSupportRequest,
    PlaceSetSupportResult,
    PumpingWitnessResult,
    ReachabilityResult,
    SiphonTrapFamilyResult,
    SiphonTrapResult,
    StateEquationResult,
    _marking_commutation_profile_output_bound,
    _marking_conflict_profile_output_bound,
)
from jacobian.math.logic.automata.petri_nets.values import (
    MAX_PETRI_MARKING,
    FiringSequence,
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
    "check_pumping_witness",
    "compute_incidence_matrix",
    "concurrent_step",
    "enabled_transitions",
    "find_minimal_siphons",
    "find_minimal_traps",
    "fire_transition",
    "marking_commutation_profile",
    "marking_conflict_profile",
    "marking_reachability",
    "petri_invariants",
    "petri_nonnegative_invariant_generators",
    "place_set_initial_marking_profile",
    "place_set_support",
    "reachability_graph",
    "replay_firing_sequence",
    "reverse_petri_net",
    "siphon_trap",
    "siphon_trap_family",
    "state_equation_target",
    "verify_enabled_transitions",
    "verify_fire_transition",
    "verify_incidence_matrix",
    "verify_invariants",
    "verify_reachability_graph",
    "verify_siphon_trap",
]

MAX_PETRI_NET_REVERSE_OUTPUT_BYTES = 10 * 1024 * 1024


def _petri_net_reverse_output_bound(net: PetriNet) -> int:
    """Compute the exact canonical JSON size without materializing the net."""

    def array_size(item_sizes: list[int]) -> int:
        return 2 + max(0, len(item_sizes) - 1) + sum(item_sizes)

    def string_size(value: str) -> int:
        size = 2  # surrounding quotes
        short_escapes = {"\b", "\t", "\n", "\f", "\r"}
        for character in value:
            codepoint = ord(character)
            if character in ('"', "\\") or character in short_escapes:
                size += 2
            elif codepoint <= 0x1F:
                size += 6  # RFC 8785 uses a \u00xx escape for other controls.
            elif 0xD800 <= codepoint <= 0xDFFF:
                # RFC 8785 cannot encode unpaired UTF-16 surrogates.
                raise OperationDomainValidationError(
                    location=("net",),
                    code="petri_net.net_axis_encoding",
                    message="place and transition IDs must be valid Unicode strings",
                )
            elif codepoint <= 0x7F:
                size += 1
            elif codepoint <= 0x7FF:
                size += 2
            elif codepoint <= 0xFFFF:
                size += 3
            else:
                size += 4
        return size

    def ids_size(ids: tuple[str, ...] | None) -> int:
        return 4 if ids is None else array_size([string_size(item) for item in ids])

    def matrix_size(matrix: tuple[tuple[int, ...], ...]) -> int:
        return array_size(
            [array_size([len(str(entry)) for entry in row]) for row in matrix]
        )

    return strict_json_object_size(
        (
            ("place_count", len(str(net.place_count))),
            ("transition_count", len(str(net.transition_count))),
            ("place_ids", ids_size(net.place_ids)),
            ("transition_ids", ids_size(net.transition_ids)),
            # Swapping the matrices preserves their combined serialized size.
            ("pre", matrix_size(net.pre)),
            ("post", matrix_size(net.post)),
        )
    )


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


def reverse_petri_net(net: PetriNet) -> PetriNet:
    """Reverse every transition by exchanging its input and output arcs."""

    admitted = _admit_net(net)
    output_bytes = _petri_net_reverse_output_bound(admitted)
    if output_bytes > MAX_PETRI_NET_REVERSE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("net",),
            code="petri_net.reverse_output_bound",
            message="reversed Petri net exceeds the serialized output bound",
        )
    # The matrices have identical shape and limits. Reversal swaps canonical
    # immutable tuples, so no expanded matrix or second validation pass is needed.
    return PetriNet.model_construct(
        place_count=admitted.place_count,
        transition_count=admitted.transition_count,
        place_ids=admitted.place_ids,
        transition_ids=admitted.transition_ids,
        pre=admitted.post,
        post=admitted.pre,
    )


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


def marking_conflict_profile(
    net: PetriNet, marking: Marking
) -> MarkingConflictProfileResult:
    """Partition individually enabled transition pairs by step compatibility.

    This is a local resource-conflict profile, not a claim about sequential
    commutation: both transitions can be individually enabled while aggregate
    simultaneous consumption is impossible.
    """

    net = _admit_net(net)
    marking = _require_marking_size(net, marking)
    enabled = tuple(_enabled_transition_indices(net, marking))
    pair_count = len(enabled) * (len(enabled) - 1) // 2
    work = pair_count * net.place_count
    if work > 150_000:
        raise OperationResourceAdmissionError(
            location=("net", "marking"),
            code="petri_net.conflict_profile_bound",
            message="pairwise marking conflict profile exceeds its work bound",
        )
    output_bound = _marking_conflict_profile_output_bound(
        net, marking, enabled_count=len(enabled), pair_count=pair_count
    )
    if output_bound > MAX_MARKING_CONFLICT_PROFILE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("net", "marking"),
            code="petri_net.conflict_profile_output_bound",
            message="marking conflict profile exceeds the serialized result bound",
        )
    jointly_enabled: list[tuple[int, int]] = []
    conflicts: list[tuple[int, int]] = []
    for left, right in combinations(enabled, 2):
        if all(
            marking.tokens[place] >= net.pre[place][left] + net.pre[place][right]
            for place in range(net.place_count)
        ):
            jointly_enabled.append((left, right))
        else:
            conflicts.append((left, right))
    return MarkingConflictProfileResult._from_kernel(
        net=net,
        marking=marking,
        enabled_transitions=enabled,
        jointly_enabled_pairs=tuple(jointly_enabled),
        conflicting_pairs=tuple(conflicts),
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


def concurrent_step(
    net: PetriNet, marking: Marking, transition_counts: tuple[int, ...]
) -> ConcurrentStepResult:
    """Fire a multiset simultaneously, requiring all consumed resources up front.

    This is step semantics: aggregate consumption is checked against the source
    marking before aggregate production is applied. It makes no claim that any
    sequential ordering of the multiset is fireable.
    """

    net = _admit_net(net)
    marking = _require_marking_size(net, marking)
    if len(transition_counts) != net.transition_count or any(
        type(count) is not int or count < 0 for count in transition_counts
    ):
        raise OperationDomainValidationError(
            location=("transition_counts",),
            code="petri_net.step_transition_axis",
            message="transition counts must be nonnegative integers on the net axis",
        )
    total = sum(transition_counts)
    if total > MAX_CONCURRENT_STEP_OCCURRENCES:
        raise OperationResourceAdmissionError(
            location=("transition_counts",),
            code="petri_net.step_occurrence_bound",
            message="simultaneous step exceeds the admitted total multiplicity",
        )
    required = tuple(
        sum(
            net.pre[place][transition] * transition_counts[transition]
            for transition in range(net.transition_count)
        )
        for place in range(net.place_count)
    )
    deficit = tuple(
        max(0, need - have) for need, have in zip(required, marking.tokens, strict=True)
    )
    if any(deficit):
        return ConcurrentStepResult(
            net=net,
            marking=marking,
            transition_counts=transition_counts,
            required=required,
            deficit=deficit,
            status="NOT_ENABLED",
        )
    target = tuple(
        marking.tokens[place]
        + sum(
            (net.post[place][transition] - net.pre[place][transition])
            * transition_counts[transition]
            for transition in range(net.transition_count)
        )
        for place in range(net.place_count)
    )
    if any(token > MAX_PETRI_MARKING for token in target):
        return ConcurrentStepResult(
            net=net,
            marking=marking,
            transition_counts=transition_counts,
            required=required,
            deficit=deficit,
            status="ESCAPES_DECLARED_ENVELOPE",
            envelope_escape=target,
        )
    return ConcurrentStepResult(
        net=net,
        marking=marking,
        transition_counts=transition_counts,
        required=required,
        deficit=deficit,
        status="FIRED",
        new_marking=_bound_marking(net, marking, target),
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
    return _replay_firing_sequence_admitted(net, marking, sequence)


def _replay_firing_sequence_admitted(
    net: PetriNet, marking: Marking, sequence: tuple[int, ...]
) -> FiringSequenceReplayResult:
    """Replay a sequence after the shared net, marking, and axis admission."""

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


def check_pumping_witness(
    net: PetriNet, marking: Marking, sequence: tuple[int, ...]
) -> PumpingWitnessResult:
    """Check a concrete replay M ->* M' with M' >= M and M' != M.

    Ordinary P/T transition maps are monotone: the same sequence remains
    enabled from any marking above M, and its final marking is shifted by the
    same nonnegative delta. Repeating strict growth therefore proves unboundedness.
    """
    net = _admit_net(net)
    marking = _require_marking_size(net, marking)
    _require_sequence_axes(net, sequence)
    replay = _replay_firing_sequence_admitted(net, marking, sequence)
    if replay.status == "BLOCKED":
        return PumpingWitnessResult(
            net=net,
            marking=marking,
            sequence=sequence,
            replay=replay,
            status="BLOCKED",
        )
    assert replay.final_marking is not None
    delta = tuple(
        final - initial
        for initial, final in zip(
            marking.tokens, replay.final_marking.tokens, strict=True
        )
    )
    grows = any(delta) and all(value >= 0 for value in delta)
    return PumpingWitnessResult(
        net=net,
        marking=marking,
        sequence=sequence,
        replay=replay,
        status="PUMPING_WITNESS" if grows else "FIRES_WITHOUT_GROWTH",
        growth=delta if grows else None,
    )


def _admit_commutation_pair(
    net: PetriNet, transitions: tuple[int, int]
) -> tuple[int, int]:
    """Validate the distinct, ordered transition pair on one admitted net."""

    if not isinstance(transitions, tuple) or len(transitions) != 2:
        raise OperationDomainValidationError(
            location=("transitions",),
            code="petri_net.commutation_transition_pair",
            message="commutation profiling requires an ordered pair of transitions",
        )
    first, second = transitions
    if (
        type(first) is not int
        or type(second) is not int
        or first == second
        or not 0 <= first < net.transition_count
        or not 0 <= second < net.transition_count
    ):
        raise OperationDomainValidationError(
            location=("transitions",),
            code="petri_net.commutation_transition_pair",
            message="commutation profiling requires two distinct transitions on the net axis",
        )
    return first, second


def marking_commutation_profile(
    net: PetriNet, marking: Marking, transitions: tuple[int, int]
) -> MarkingCommutationProfileResult:
    """Replay both orders of a distinct transition pair from one marking.

    Each order is an exact two-step sequential replay. Simultaneous enabledness
    is a separate operation, and neither result says anything about global
    confluence or other reachable markings.
    """

    admitted_net = _admit_net(net)
    admitted_marking = _require_marking_size(admitted_net, marking)
    first, second = _admit_commutation_pair(admitted_net, transitions)
    output_bound = _marking_commutation_profile_output_bound(
        admitted_net, admitted_marking
    )
    if output_bound > MAX_MARKING_COMMUTATION_PROFILE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("net", "marking"),
            code="petri_net.commutation_profile_output_bound",
            message="marking commutation profile exceeds the serialized output bound",
        )
    # Two length-two replays each compare deficits and update a marking per
    # selected transition, then compute one place-by-transition residual.
    work_bound = 2 * (
        4 * admitted_net.place_count
        + admitted_net.place_count * admitted_net.transition_count
    )
    if work_bound > MAX_MARKING_COMMUTATION_PROFILE_WORK:
        raise OperationResourceAdmissionError(
            location=("net", "transitions"),
            code="petri_net.commutation_profile_work_bound",
            message="marking commutation profile exceeds the replay work bound",
        )

    first_then_second = _replay_firing_sequence_admitted(
        admitted_net, admitted_marking, (first, second)
    )
    second_then_first = _replay_firing_sequence_admitted(
        admitted_net, admitted_marking, (second, first)
    )
    both_orders_fire = first_then_second.status == second_then_first.status == "FIRES"
    same_target = bool(
        both_orders_fire
        and first_then_second.final_marking == second_then_first.final_marking
    )
    return MarkingCommutationProfileResult._from_kernel(
        first_then_second=first_then_second,
        second_then_first=second_then_first,
        both_orders_fire=both_orders_fire,
        same_target=same_target,
    )


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


def state_equation_target(
    net: PetriNet, marking: Marking, transition_counts: tuple[int, ...]
) -> StateEquationResult:
    """Compute M0 + (Post - Pre)y over Z, without asserting reachability."""
    net = _admit_net(net)
    marking = _require_marking_size(net, marking)
    if len(transition_counts) != net.transition_count or any(
        type(count) is not int or count < 0 for count in transition_counts
    ):
        raise OperationDomainValidationError(
            location=("transition_counts",),
            code="petri_net.state_equation_transition_axis",
            message="transition counts must be nonnegative integers on the net axis",
        )
    if sum(transition_counts) > MAX_STATE_EQUATION_OCCURRENCES:
        raise OperationResourceAdmissionError(
            location=("transition_counts",),
            code="petri_net.state_equation_occurrence_bound",
            message="total transition count exceeds the admitted bound",
        )
    # At most 64 places, 64 transitions, 1000 occurrences, and arc weights
    # of 1000 yield target magnitudes below 64,001,000 and a tiny exact result.
    target = tuple(
        marking.tokens[place]
        + sum(
            (net.post[place][transition] - net.pre[place][transition])
            * transition_counts[transition]
            for transition in range(net.transition_count)
        )
        for place in range(net.place_count)
    )
    return StateEquationResult(
        net=net,
        marking=marking,
        transition_counts=transition_counts,
        target=target,
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


def marking_reachability(
    net: PetriNet,
    initial_marking: Marking,
    target_marking: Marking,
    max_states: int = 10000,
) -> MarkingReachabilityResult:
    """Search for one executable sequence to a target marking.

    BFS retains only visited markings and predecessor pairs. A found sequence
    is an exact positive witness even if another branch was cut. A negative
    conclusion is returned only when the queue is exhausted without crossing
    either the state or token envelope.
    """

    net = _admit_net(net)
    initial_marking = _require_marking_size(net, initial_marking)
    target_marking = _require_marking_size(net, target_marking)
    require_reachability_bounds(net, max_states)

    initial = tuple(initial_marking.tokens)
    target = tuple(target_marking.tokens)
    if initial == target:
        return MarkingReachabilityResult(
            net=net,
            initial_marking=initial_marking,
            target_marking=target_marking,
            max_states=max_states,
            status="REACHABLE",
            sequence=FiringSequence(transitions=()),
            explored_state_count=1,
        )
    states: list[tuple[int, ...]] = [initial]
    state_index = {initial: 0}
    # Each entry stores (parent state index, transition fired).
    predecessor: list[tuple[int, int] | None] = [None]
    queue: deque[int] = deque([0])
    incomplete_reasons: set[str] = set()

    def witness(state: int) -> tuple[int, ...]:
        transitions: list[int] = []
        while predecessor[state] is not None:
            parent, transition = predecessor[state]
            transitions.append(transition)
            state = parent
        transitions.reverse()
        return tuple(transitions)

    while queue:
        source_index = queue.popleft()
        source = _bound_marking(net, initial_marking, states[source_index])
        for transition in _enabled_transition_indices(net, source):
            _, successor = _fire_transition_tokens(net, source, transition)
            if any(token > MAX_PETRI_MARKING for token in successor):
                incomplete_reasons.add("MARKING_LIMIT")
                continue
            target_index = state_index.get(successor)
            if target_index is None:
                if len(states) >= max_states:
                    incomplete_reasons.add("STATE_LIMIT")
                    continue
                target_index = len(states)
                state_index[successor] = target_index
                states.append(successor)
                predecessor.append((source_index, transition))
                queue.append(target_index)
            if successor == target:
                transitions = witness(target_index)
                if len(transitions) > MAX_FIRING_SEQUENCE_LENGTH:
                    incomplete_reasons.add("SEQUENCE_LIMIT")
                    return MarkingReachabilityResult(
                        net=net,
                        initial_marking=initial_marking,
                        target_marking=target_marking,
                        max_states=max_states,
                        status="INCOMPLETE",
                        sequence=None,
                        explored_state_count=len(states),
                        incomplete_reasons=tuple(sorted(incomplete_reasons)),
                    )
                return MarkingReachabilityResult(
                    net=net,
                    initial_marking=initial_marking,
                    target_marking=target_marking,
                    max_states=max_states,
                    status="REACHABLE",
                    sequence=FiringSequence(transitions=transitions),
                    explored_state_count=len(states),
                )

    return MarkingReachabilityResult(
        net=net,
        initial_marking=initial_marking,
        target_marking=target_marking,
        max_states=max_states,
        status="INCOMPLETE" if incomplete_reasons else "UNREACHABLE",
        explored_state_count=len(states),
        incomplete_reasons=tuple(sorted(incomplete_reasons)),
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


def siphon_trap_family(net: PetriNet) -> SiphonTrapFamilyResult:
    """Enumerate every nonempty siphon and trap by exact support semantics.

    The complete subset scan is intentionally limited by both its transition
    work and a conservative serialized-family upper bound before enumeration.
    """
    net = _admit_net(net)
    places = net.place_count
    subsets = (1 << places) - 1
    work = places * net.transition_count + subsets * (
        2 * net.transition_count + 3 * places + 1
    )
    if work > MAX_SIPHON_TRAP_WORK:
        raise OperationResourceAdmissionError(
            location=("net",),
            code="petri_net.siphon_trap_family_work_bound",
            message="complete siphon/trap subset scan exceeds the admitted work bound",
        )
    index_digits = max(1, len(str(max(0, places - 1))))
    bytes_per_subset = 32 + places * (index_digits + 1)
    output_bytes = (
        2 * subsets * bytes_per_subset
        + len(net.model_dump_json().encode("utf-8"))
        + 256
    )
    if output_bytes > MAX_SIPHON_TRAP_FAMILY_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("net",),
            code="petri_net.siphon_trap_family_output_bound",
            message="complete siphon/trap families exceed the admitted output bound",
        )

    input_support = [0] * net.transition_count
    output_support = [0] * net.transition_count
    for place in range(places):
        bit = 1 << place
        for transition in range(net.transition_count):
            if net.pre[place][transition] > 0:
                input_support[transition] |= bit
            if net.post[place][transition] > 0:
                output_support[transition] |= bit

    siphons: list[PetriPlaceSubset] = []
    traps: list[PetriPlaceSubset] = []
    for size in range(1, places + 1):
        for members in combinations(range(places), size):
            mask = sum(1 << place for place in members)
            is_siphon = all(
                not (mask & output) or (mask & source)
                for source, output in zip(input_support, output_support, strict=True)
            )
            is_trap = all(
                not (mask & source) or (mask & output)
                for source, output in zip(input_support, output_support, strict=True)
            )
            if is_siphon:
                siphons.append(PetriPlaceSubset(places=members))
            if is_trap:
                traps.append(PetriPlaceSubset(places=members))
    return SiphonTrapFamilyResult(net=net, siphons=tuple(siphons), traps=tuple(traps))


def place_set_support(net: PetriNet, places: PetriPlaceSubset) -> PlaceSetSupportResult:
    """Return the exact producer/consumer profile of a declared place set.

    Here a transition produces into a set when it has a positive Post arc to
    any place in the set, and consumes from it when it has a positive Pre arc
    from any place in the set. Thus the siphon predicate is
    ``producers_into <= consumers_from`` and the trap predicate is the reverse
    inclusion.
    """
    net = _admit_net(net)
    request = PlaceSetSupportRequest.model_validate(
        {"net": net, "places": places}, strict=True
    )
    return _place_set_support_admitted(net, request.places)


def _place_set_support_admitted(
    net: PetriNet, places: PetriPlaceSubset
) -> PlaceSetSupportResult:
    """Compute a support profile after net and subset axes are admitted."""

    selected = set(places.places)
    producers = tuple(
        transition
        for transition in range(net.transition_count)
        if any(net.post[place][transition] > 0 for place in selected)
    )
    consumers = tuple(
        transition
        for transition in range(net.transition_count)
        if any(net.pre[place][transition] > 0 for place in selected)
    )
    siphon_offenders = tuple(t for t in producers if t not in consumers)
    trap_offenders = tuple(t for t in consumers if t not in producers)
    return PlaceSetSupportResult(
        net=net,
        places=places,
        producers_into=producers,
        consumers_from=consumers,
        siphon_offenders=siphon_offenders,
        trap_offenders=trap_offenders,
        is_siphon=not siphon_offenders,
        is_trap=not trap_offenders,
    )


def place_set_initial_marking_profile(
    net: PetriNet,
    places: PetriPlaceSubset,
    marking: Marking,
) -> PlaceSetInitialMarkingProfileResult:
    """Return selected token count and applicable siphon/trap persistence laws."""

    net = _admit_net(net)
    marking = _require_marking_size(net, marking)
    if not isinstance(places, PetriPlaceSubset):
        raise OperationDomainValidationError(
            location=("places",),
            code="petri_net.place_subset_type",
            message="places must be a PetriPlaceSubset value",
        )
    try:
        places = PetriPlaceSubset.model_validate(places.model_dump(), strict=True)
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("places",),
            code="petri_net.place_subset_shape",
            message="places must be a canonical PetriPlaceSubset value",
        ) from exc
    if any(place >= net.place_count for place in places.places):
        raise OperationDomainValidationError(
            location=("places",),
            code="petri_net.place_axis",
            message="subset must use the net place axis",
        )

    support = _place_set_support_admitted(net, places)
    total = sum(marking.tokens[place] for place in places.places)
    implications: list[
        Literal["EMPTY_SIPHON_REMAINS_EMPTY", "MARKED_TRAP_REMAINS_MARKED"]
    ] = []
    if support.is_siphon and total == 0:
        implications.append("EMPTY_SIPHON_REMAINS_EMPTY")
    if support.is_trap and total > 0:
        implications.append("MARKED_TRAP_REMAINS_MARKED")
    return PlaceSetInitialMarkingProfileResult(
        net=net,
        places=places,
        marking=marking,
        selected_token_total=total,
        support_profile=support,
        preservation_implications=tuple(implications),
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


def petri_invariants(net: PetriNet) -> PetriInvariantsResult:
    """Compute P-invariants and T-invariants as exact integer modules.

    P-invariants span ``ker_Z(C^T)`` and T-invariants span ``ker_Z(C)``
    for the incidence matrix ``C``. Kernel bases come from the certified
    Smith owner kernel, canonicalized through the Hermite owner kernel;
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
    return PetriInvariantsResult._from_kernel(
        net=net,
        incidence=incidence_matrix,
        incidence_rank=rank,
        p_invariants=p_basis,
        t_invariants=t_basis,
    )


def _nonnegative_kernel_admission(
    matrix: tuple[tuple[int, ...], ...], dimension: int
) -> tuple[int, int, int]:
    """Return a sound enumeration/work/output bound for one integer kernel."""
    row_count = len(matrix)
    if dimension == 0:
        return (0, 0, 0)
    if row_count == 0 or not any(any(row) for row in matrix):
        return (0, dimension * dimension, dimension * (dimension + 2))
    # A nonzero 1-dimensional vector lies in the kernel only when every
    # coefficient vanishes, handled above.
    if dimension == 1:
        return (0, 0, 0)
    delta = max(abs(value) for row in matrix for value in row)
    # Steinitz ordering: a conformally indecomposable zero-sum sequence of
    # integer columns has distinct partial sums in [-m*Delta,m*Delta]^m.
    # Thus every orthant Graver element, and hence each Hilbert basis vector,
    # has l1 norm at most this bound.
    height = (2 * row_count * delta + 1) ** row_count
    candidates = comb(height + dimension, dimension)
    # Bound testing each candidate against all earlier kernel candidates,
    # including exact matrix products and coordinate comparisons.
    work = candidates * candidates * max(1, dimension) + candidates * max(
        1, row_count * dimension
    )
    output_bytes = candidates * dimension * (len(str(height)) + 2)
    return height, work, output_bytes


def _weak_compositions(total: int, length: int):
    """Generate weak compositions in lexicographic order."""
    if length == 0:
        if total == 0:
            yield ()
        return
    if length == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for suffix in _weak_compositions(total - first, length - 1):
            yield (first, *suffix)


def _nonnegative_integer_kernel_generators(
    matrix: tuple[tuple[int, ...], ...], dimension: int, height: int
) -> tuple[tuple[int, ...], ...]:
    """Enumerate all bounded kernel vectors and retain the indecomposables."""
    if dimension == 0:
        return ()
    if not matrix or not any(any(row) for row in matrix):
        return tuple(
            tuple(1 if coordinate == index else 0 for coordinate in range(dimension))
            for index in range(dimension)
        )
    if dimension == 1:
        return ()
    kernel_vectors: list[tuple[int, ...]] = []
    generators: list[tuple[int, ...]] = []
    for norm in range(1, height + 1):
        for vector in _weak_compositions(norm, dimension):
            if any(sum(a * x for a, x in zip(row, vector, strict=True)) for row in matrix):
                continue
            kernel_vectors.append(vector)
            if not any(
                all(part <= value for part, value in zip(previous, vector, strict=True))
                for previous in kernel_vectors[:-1]
            ):
                generators.append(vector)
    return tuple(sorted(generators))


def petri_nonnegative_invariant_generators(
    net: PetriNet,
) -> PetriNonnegativeInvariantResult:
    """Compute Hilbert bases for the nonnegative P- and T-invariant monoids.

    This is complete within the admitted finite enumeration bound. The result
    describes algebraic kernel vectors only; in particular, a T-invariant
    does not imply that its transition multiset is fireable from any marking.
    """
    net = _admit_net(net)
    incidence = tuple(
        tuple(
            net.post[place][transition] - net.pre[place][transition]
            for transition in range(net.transition_count)
        )
        for place in range(net.place_count)
    )
    transposed = tuple(
        tuple(incidence[place][transition] for place in range(net.place_count))
        for transition in range(net.transition_count)
    )
    t_bound = _nonnegative_kernel_admission(incidence, net.transition_count)
    p_bound = _nonnegative_kernel_admission(transposed, net.place_count)
    total_work = t_bound[1] + p_bound[1]
    total_output = (
        t_bound[2]
        + p_bound[2]
        + _petri_net_reverse_output_bound(net)
        + 512
    )
    if total_work > MAX_PETRI_NONNEGATIVE_INVARIANT_WORK:
        raise OperationResourceAdmissionError(
            location=("net",),
            code="petri_net.nonnegative_invariant_work",
            message="nonnegative invariant Hilbert-basis search exceeds its exact work bound",
        )
    if total_output > MAX_PETRI_NONNEGATIVE_INVARIANT_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("net",),
            code="petri_net.nonnegative_invariant_output",
            message="nonnegative invariant Hilbert-basis output exceeds its bound",
        )
    return PetriNonnegativeInvariantResult._from_kernel(
        net=net,
        p_generators=_nonnegative_integer_kernel_generators(
            transposed, net.place_count, p_bound[0]
        ),
        t_generators=_nonnegative_integer_kernel_generators(
            incidence, net.transition_count, t_bound[0]
        ),
    )


def verify_invariants(claim: PetriInvariantsResult) -> bool:
    """Verify P/T-invariant bases against their retained net."""

    try:
        return petri_invariants(claim.net) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
