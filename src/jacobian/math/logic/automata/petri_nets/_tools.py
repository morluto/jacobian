"""Petri net operation declarations."""

from typing import Any

from jacobian.catalog.models import (
    MathTool,
    OperationExample,
)
from jacobian.math.logic.automata.petri_nets._models import (
    ConcurrentStepRequest,
    ConcurrentStepResult,
    EnabledTransitionsRequest,
    EnabledTransitionsResult,
    FireTransitionRequest,
    FireTransitionResult,
    FiringSequenceReplayRequest,
    FiringSequenceReplayResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
    MarkingCommutationProfileRequest,
    MarkingCommutationProfileResult,
    MarkingConflictProfileRequest,
    MarkingConflictProfileResult,
    MarkingReachabilityRequest,
    MarkingReachabilityResult,
    PetriInvariantsRequest,
    PetriInvariantsResult,
    PetriNetMatricesRequest,
    PetriNetMatricesResult,
    PetriNetRelabelingRequest,
    PetriNetRelabelingResult,
    PlaceSetInitialMarkingProfileRequest,
    PlaceSetInitialMarkingProfileResult,
    PlaceSetSupportRequest,
    PlaceSetSupportResult,
    PumpingWitnessRequest,
    PumpingWitnessResult,
    ReachabilityRequest,
    ReachabilityResult,
    ReachabilityTerminalSCCProfileRequest,
    ReachabilityTerminalSCCProfileResult,
    ReachableDeadMarkingsRequest,
    ReachableDeadMarkingsResult,
    SiphonTrapFamilyRequest,
    SiphonTrapFamilyResult,
    SiphonTrapRequest,
    SiphonTrapResult,
    StateEquationRequest,
    StateEquationResult,
)
from jacobian.math.logic.automata.petri_nets.operations import (
    check_pumping_witness,
    compute_incidence_matrix,
    concurrent_step,
    enabled_transitions,
    fire_transition,
    marking_commutation_profile,
    marking_conflict_profile,
    marking_reachability,
    petri_invariants,
    petri_net_matrices,
    place_set_initial_marking_profile,
    place_set_support,
    reachability_graph,
    reachability_terminal_scc_profile,
    reachable_dead_markings,
    relabel_petri_net,
    replay_firing_sequence,
    reverse_petri_net,
    siphon_trap,
    siphon_trap_family,
    state_equation_target,
)
from jacobian.math.logic.automata.petri_nets.values import PetriNet


def compute_enabled_transitions(
    request: EnabledTransitionsRequest,
) -> EnabledTransitionsResult:
    return enabled_transitions(request.net, request.marking)


def compute_petri_net_relabeling(
    request: PetriNetRelabelingRequest,
) -> PetriNetRelabelingResult:
    return relabel_petri_net(request)


def compute_marking_conflict_profile(
    request: MarkingConflictProfileRequest,
) -> MarkingConflictProfileResult:
    return marking_conflict_profile(request.net, request.marking)


def compute_marking_commutation_profile(
    request: MarkingCommutationProfileRequest,
) -> MarkingCommutationProfileResult:
    return marking_commutation_profile(
        request.net, request.marking, request.transitions
    )


def compute_fire_transition(request: FireTransitionRequest) -> FireTransitionResult:
    return fire_transition(request.net, request.marking, request.transition)


def compute_concurrent_step(request: ConcurrentStepRequest) -> ConcurrentStepResult:
    return concurrent_step(request.net, request.marking, request.transition_counts)


def compute_incidence(request: IncidenceMatrixRequest) -> IncidenceMatrixResult:
    return compute_incidence_matrix(request.net)


def compute_state_equation(request: StateEquationRequest) -> StateEquationResult:
    return state_equation_target(
        request.net, request.marking, request.transition_counts
    )


def compute_reachability(request: ReachabilityRequest) -> ReachabilityResult:
    return reachability_graph(request.net, request.initial_marking, request.max_states)


def compute_reachable_dead_markings(
    request: ReachableDeadMarkingsRequest,
) -> ReachableDeadMarkingsResult:
    return reachable_dead_markings(
        request.net, request.initial_marking, request.max_states
    )


def compute_reachability_terminal_scc_profile(
    request: ReachabilityTerminalSCCProfileRequest,
) -> ReachabilityTerminalSCCProfileResult:
    return reachability_terminal_scc_profile(request.source_graph)


def compute_marking_reachability(
    request: MarkingReachabilityRequest,
) -> MarkingReachabilityResult:
    return marking_reachability(
        request.net,
        request.initial_marking,
        request.target_marking,
        request.max_states,
    )


def compute_siphon_trap(request: SiphonTrapRequest) -> SiphonTrapResult:
    return siphon_trap(request.net)


def compute_siphon_trap_family(
    request: SiphonTrapFamilyRequest,
) -> SiphonTrapFamilyResult:
    return siphon_trap_family(request.net)


def compute_petri_invariants(request: PetriInvariantsRequest) -> PetriInvariantsResult:
    return petri_invariants(request.net)


def compute_petri_net_matrices(
    request: PetriNetMatricesRequest,
) -> PetriNetMatricesResult:
    return petri_net_matrices(request.net)


def compute_place_set_support(request: PlaceSetSupportRequest) -> PlaceSetSupportResult:
    return place_set_support(request.net, request.places)


def compute_place_set_initial_marking_profile(
    request: PlaceSetInitialMarkingProfileRequest,
) -> PlaceSetInitialMarkingProfileResult:
    return place_set_initial_marking_profile(
        request.net, request.places, request.marking
    )


def compute_firing_sequence_replay(
    request: FiringSequenceReplayRequest,
) -> FiringSequenceReplayResult:
    return replay_firing_sequence(request.net, request.marking, request.sequence)


def compute_pumping_witness(request: PumpingWitnessRequest) -> PumpingWitnessResult:
    return check_pumping_witness(request.net, request.marking, request.sequence)


# Simple net: 2 places, 2 transitions
# t0: p0 -> p1 (pre=[[1,0],[0,0]], post=[[0,0],[0,1]])
# t1: p1 -> p0 (pre=[[0,0],[0,1]], post=[[1,0],[0,0]])
_NET = {
    "net": {
        "place_count": 2,
        "transition_count": 2,
        "pre": [[1, 0], [0, 0]],
        "post": [[0, 0], [0, 1]],
    },
}

_NET2 = {
    "net": {
        "place_count": 2,
        "transition_count": 2,
        "pre": [[1, 0], [0, 0]],
        "post": [[0, 0], [0, 1]],
    },
}

# Producer/consumer net: t0 produces a token into p0, t1 consumes it.
# C = [[1, -1]]: T-invariants span (1, 1); no nonzero P-invariant exists.
_PRODUCER_CONSUMER_NET = {
    "place_count": 1,
    "transition_count": 2,
    "pre": [[0, 1]],
    "post": [[1, 0]],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="petri_net.matrices.compute",
        title="Compute Petri-net pre, post, and incidence matrices",
        description="Return exact precondition and postcondition matrices and their incidence difference C = Post - Pre. The source net retains the place and transition axes.",
        request_type=PetriNetMatricesRequest,
        result_type=PetriNetMatricesResult,
        run=compute_petri_net_matrices,
        tags=("petri-net", "matrices", "exact"),
        discovery_terms=("pre-incidence", "post-incidence", "incidence matrix"),
        examples=(
            OperationExample(
                name="weighted_pre_post_matrices",
                description="Return all three matrices for a two-place weighted net.",
                input={
                    "net": {
                        "place_count": 2,
                        "transition_count": 1,
                        "pre": [[2], [0]],
                        "post": [[0], [3]],
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.relabel.compute",
        title="Relabel Petri-net place and transition axes",
        description=(
            "Permute both ordered axes by explicit source-to-target bijections. "
            "Arc matrices and optional element IDs follow those maps, and the "
            "result retains both nets and the exact isomorphism maps."
        ),
        request_type=PetriNetRelabelingRequest,
        result_type=PetriNetRelabelingResult,
        run=compute_petri_net_relabeling,
        tags=("petri-net", "net-transform", "isomorphism", "exact"),
        discovery_terms=(
            "Petri net isomorphism",
            "relabel places and transitions",
            "permute Petri net axes",
        ),
        examples=(
            OperationExample(
                name="weighted_axis_relabeling",
                description=(
                    "Swap both axes of a weighted net while retaining the "
                    "source-to-target bijections."
                ),
                input={
                    "net": {
                        "place_count": 2,
                        "transition_count": 2,
                        "place_ids": ["buffer", "output"],
                        "transition_ids": ["load", "unload"],
                        "pre": [[2, 0], [0, 1]],
                        "post": [[0, 1], [1, 0]],
                    },
                    "place_source_to_target": [1, 0],
                    "transition_source_to_target": [1, 0],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.reverse.compute",
        title="Reverse a Petri net",
        description=(
            "Construct the net with the same ordered place and transition axes "
            "and with every transition's input and output arc matrices swapped. "
            "A firing in one direction is reversed by the same transition from "
            "the resulting marking; this operation makes no reachability claim."
        ),
        request_type=PetriNet,
        result_type=PetriNet,
        run=reverse_petri_net,
        tags=("petri-net", "net-transform", "exact"),
        discovery_terms=("reverse Petri net", "reverse transition arcs"),
        examples=(
            OperationExample(
                name="reverse_one_transition",
                description="Exchange the input and output arcs of a transition.",
                input={
                    "place_count": 2,
                    "transition_count": 1,
                    "place_ids": ["p", "q"],
                    "transition_ids": ["move"],
                    "pre": [[1], [0]],
                    "post": [[0], [1]],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.marking.conflict_profile.compute",
        title="Profile pairwise Petri-net resource conflicts",
        description=(
            "Partition every pair of individually enabled transitions into "
            "pairs that are jointly enabled as one simultaneous step and "
            "pairs whose aggregate input demand conflicts at the source marking. "
            "This describes step compatibility, not sequential commutation."
        ),
        request_type=MarkingConflictProfileRequest,
        result_type=MarkingConflictProfileResult,
        run=compute_marking_conflict_profile,
        tags=("petri-net", "concurrency", "conflict", "exact"),
        discovery_terms=(
            "Petri transition conflict",
            "step compatibility",
            "joint enabledness",
        ),
        examples=(
            OperationExample(
                name="two_consumers_conflict_for_one_token",
                description="Each transition is enabled alone but they cannot consume the same token simultaneously.",
                input={
                    "net": {
                        "place_count": 1,
                        "transition_count": 2,
                        "pre": [[1, 1]],
                        "post": [[0, 0]],
                    },
                    "marking": {"tokens": [1]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.marking.commutation_profile.compute",
        title="Replay both orders of two Petri-net transitions",
        description=(
            "Replay the two distinct selected transitions in both sequential "
            "orders. Return the exact prefix/final markings or first-blocking "
            "deficits for each order, together with whether both orders fire "
            "and reach the same target. This local sequential diamond is "
            "separate from simultaneous-step conflict and makes no global "
            "confluence claim."
        ),
        request_type=MarkingCommutationProfileRequest,
        result_type=MarkingCommutationProfileResult,
        run=compute_marking_commutation_profile,
        tags=("petri-net", "concurrency", "commutation", "exact"),
        discovery_terms=(
            "Petri transition commutation",
            "sequential firing order",
            "local diamond",
        ),
        examples=(
            OperationExample(
                name="only_produce_then_consume_fires",
                description=(
                    "Transition 1 produces the token that transition 0 needs, "
                    "so only one sequential order fires."
                ),
                input={
                    "net": {
                        "place_count": 1,
                        "transition_count": 2,
                        "pre": [[1, 0]],
                        "post": [[0, 1]],
                    },
                    "marking": {"tokens": [0]},
                    "transitions": [0, 1],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.marking.concurrent_step.compute",
        title="Fire a simultaneous Petri-net transition step",
        description=(
            "Apply a transition multiset simultaneously. All aggregate input "
            "tokens are required in the source marking before any production; "
            "the result is distinct from sequential firing-sequence replay."
        ),
        request_type=ConcurrentStepRequest,
        result_type=ConcurrentStepResult,
        run=compute_concurrent_step,
        tags=("petri-net", "concurrency", "step-semantics", "exact"),
        discovery_terms=(
            "simultaneous firing",
            "concurrent step",
            "transition multiset",
        ),
        examples=(
            OperationExample(
                name="two_consumers_share_one_token",
                description="The simultaneous pair is blocked although each transition is individually enabled.",
                input={
                    "net": {
                        "place_count": 1,
                        "transition_count": 2,
                        "pre": [[1, 1]],
                        "post": [[0, 0]],
                    },
                    "marking": {"tokens": [1]},
                    "transition_counts": [1, 1],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.enabled_transitions.compute",
        title="Find enabled transitions in a Petri net",
        description="Return the indices of all transitions enabled at the given marking, "
        "where a transition is enabled iff every place has enough tokens for "
        "its pre-condition.",
        request_type=EnabledTransitionsRequest,
        result_type=EnabledTransitionsResult,
        run=compute_enabled_transitions,
        tags=("petri-net", "enabled", "exact"),
        examples=(
            OperationExample(
                name="simple_net",
                description="Enabled transitions of a 2-place, 2-transition net.",
                input={
                    "net": _NET["net"],
                    "marking": {"tokens": [2, 0]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.fire_transition.compute",
        title="Fire one transition in a Petri net",
        description="Fire a single transition at the given marking and return whether it "
        "succeeded and the resulting marking. If the transition is not "
        "enabled, it does not fire.",
        request_type=FireTransitionRequest,
        result_type=FireTransitionResult,
        run=compute_fire_transition,
        tags=("petri-net", "firing", "exact"),
        examples=(
            OperationExample(
                name="fire_t0",
                description="Fire transition 0 from marking [2, 0].",
                input={
                    "net": _NET["net"],
                    "marking": {"tokens": [2, 0]},
                    "transition": 0,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.incidence_matrix.compute",
        title="Compute the incidence matrix of a Petri net",
        description="Return the incidence matrix C = Post - Pre for the given Petri net.",
        request_type=IncidenceMatrixRequest,
        result_type=IncidenceMatrixResult,
        run=compute_incidence,
        tags=("petri-net", "incidence", "exact"),
        examples=(
            OperationExample(
                name="simple_net_incidence",
                description="Incidence matrix of a 2-place, 2-transition net.",
                input={"net": _NET2["net"]},
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.state_equation.target.compute",
        title="Compute a Petri net state-equation target",
        description=(
            "Compute M0 + (Post - Pre)y over the integers for a nonnegative "
            "transition-count vector. This formal target does not assert that "
            "the target is a reachable marking or that the counts are fireable."
        ),
        request_type=StateEquationRequest,
        result_type=StateEquationResult,
        run=compute_state_equation,
        tags=("petri-net", "state-equation", "incidence", "exact"),
        examples=(
            OperationExample(
                name="formal_target_without_enabled_sequence",
                description=(
                    "The count vector has a valid formal target although neither "
                    "transition is enabled at the empty source marking."
                ),
                input={
                    "net": {
                        "place_count": 2,
                        "transition_count": 2,
                        "pre": [[1, 0], [0, 1]],
                        "post": [[0, 1], [1, 0]],
                    },
                    "marking": {"tokens": [0, 0]},
                    "transition_counts": [1, 1],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.reachability_graph.compute",
        title="Compute the bounded reachability graph of a Petri net",
        description="Return the bounded reachability graph from an initial marking via "
        "BFS, including all reachable markings and firing edges. The graph "
        "is truncated at max_states to bound the state space.",
        request_type=ReachabilityRequest,
        result_type=ReachabilityResult,
        run=compute_reachability,
        tags=("petri-net", "reachability", "exact"),
        examples=(
            OperationExample(
                name="simple_net_reachability",
                description="Reachability graph of a simple 2-place net.",
                input={
                    "net": _NET["net"],
                    "initial_marking": {"tokens": [1, 0]},
                    "max_states": 100,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.reachable_dead_markings.compute",
        title="Find bounded reachable dead markings of a Petri net",
        description=(
            "Explore the bounded reachable state space and list discovered "
            "markings with no enabled transition. The deadness test uses "
            "transition enabledness, so omitted edges at an exploration limit "
            "cannot create false dead markings. `truncated` reports whether "
            "the list covers the entire reachable set."
        ),
        request_type=ReachableDeadMarkingsRequest,
        result_type=ReachableDeadMarkingsResult,
        run=compute_reachable_dead_markings,
        tags=("petri-net", "reachability", "dead-marking", "exact"),
        discovery_terms=("reachable dead markings", "Petri net deadlock markings"),
        examples=(
            OperationExample(
                name="one_token_consumed_to_dead_marking",
                description="The sole firing reaches the empty, dead marking.",
                input={
                    "net": {
                        "place_count": 1,
                        "transition_count": 1,
                        "pre": [[1]],
                        "post": [[0]],
                    },
                    "initial_marking": {"tokens": [1]},
                    "max_states": 8,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.reachability.terminal_scc_profile.compute",
        title="Profile terminal components of a bounded Petri reachability graph",
        description=(
            "Return the sink strongly connected components of the supplied exact "
            "reachability graph as state-index sets. If its exploration was "
            "truncated, terminality applies only to the represented partial graph "
            "and says nothing about omitted successors or full-net recurrence."
        ),
        request_type=ReachabilityTerminalSCCProfileRequest,
        result_type=ReachabilityTerminalSCCProfileResult,
        run=compute_reachability_terminal_scc_profile,
        tags=("petri-net", "reachability", "strongly-connected-components", "exact"),
        discovery_terms=("terminal SCC", "sink strongly connected components"),
        examples=(
            OperationExample(
                name="one_state_recurrent_self_loop",
                description="A complete one-state graph with one self-loop is terminal.",
                input={
                    "source_graph": {
                        "net": {
                            "place_count": 1,
                            "transition_count": 1,
                            "pre": [[1]],
                            "post": [[1]],
                        },
                        "initial_marking": {"tokens": [1]},
                        "max_states": 8,
                        "states": [
                            {
                                "state_index": 0,
                                "place_axis": [0],
                                "marking": {"tokens": [1]},
                            }
                        ],
                        "edges": [
                            {"source_state": 0, "transition": 0, "target_state": 0}
                        ],
                        "truncated": False,
                    }
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.marking.reachability.compute",
        title="Search Petri-net reachability to a target marking",
        description=(
            "Search from an initial marking for an executable transition-sequence "
            "witness to a target. REACHABLE carries a sequence that can be replayed; "
            "UNREACHABLE is returned only after exhaustive closure. A state, token, "
            "or witness-length envelope cut returns INCOMPLETE. No result claims a "
            "shortest sequence."
        ),
        request_type=MarkingReachabilityRequest,
        result_type=MarkingReachabilityResult,
        run=compute_marking_reachability,
        tags=("petri-net", "marking", "reachability", "firing-sequence", "exact"),
        discovery_terms=(
            "target marking reachable",
            "reachability witness",
            "firing sequence to marking",
        ),
        examples=(
            OperationExample(
                name="target_requires_two_firings",
                description=(
                    "Return a transition sequence from the initial to the target "
                    "marking in a two-place cycle."
                ),
                input={
                    "net": _NET["net"],
                    "initial_marking": {"tokens": [1, 0]},
                    "target_marking": {"tokens": [0, 1]},
                    "max_states": 100,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.firing_sequence.replay.compute",
        title="Replay a bounded Petri-net firing sequence",
        description="Fire one bounded transition sequence step by step from a source "
        "marking. FIRES returns every prefix marking, the final marking, the "
        "Parikh vector, and a zero state-equation residual witnessing "
        "M_k = M_0 + C x_k. BLOCKED returns the first blocked index with the "
        "source prefix marking and the complete place-deficit profile. A "
        "later transition cannot repair an earlier disabled one.",
        request_type=FiringSequenceReplayRequest,
        result_type=FiringSequenceReplayResult,
        run=compute_firing_sequence_replay,
        tags=("petri-net", "firing-sequence", "replay", "exact"),
        discovery_terms=("firing sequence", "sequence replay", "parikh vector"),
        examples=(
            OperationExample(
                name="two_step_replay",
                description="Replay transition 0 twice from marking [2, 0].",
                input={
                    "net": _NET["net"],
                    "marking": {"tokens": [2, 0]},
                    "sequence": [0, 0],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.firing_sequence.pumping_witness.check",
        title="Check a Petri-net pumping witness",
        description=(
            "Replay a supplied ordinary P/T firing sequence. PUMPING_WITNESS "
            "returns a nonzero nonnegative marking increase; monotonicity then "
            "allows the same sequence to repeat indefinitely, proving unboundedness. "
            "BLOCKED identifies a disabled prefix, and FIRES_WITHOUT_GROWTH makes "
            "no boundedness or unboundedness claim."
        ),
        request_type=PumpingWitnessRequest,
        result_type=PumpingWitnessResult,
        run=compute_pumping_witness,
        tags=("petri-net", "firing-sequence", "unboundedness-witness", "exact"),
        discovery_terms=("Petri net unboundedness witness", "pumping firing sequence"),
        examples=(
            OperationExample(
                name="repeatable_token_production",
                description="A source transition adds one token on each repetition.",
                input={
                    "net": {
                        "place_count": 1,
                        "transition_count": 1,
                        "pre": [[0]],
                        "post": [[1]],
                    },
                    "marking": {"tokens": [0]},
                    "sequence": [0],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.invariants.compute",
        title="Compute P-invariants and T-invariants of a Petri net",
        description="Return exact integer bases for the P-invariants "
        "(left kernel of the incidence matrix) and T-invariants (right "
        "kernel) with the incidence rank profile. Kernel bases come from "
        "the Smith owner kernel and are canonicalized through the "
        "Hermite owner kernel.",
        request_type=PetriInvariantsRequest,
        result_type=PetriInvariantsResult,
        run=compute_petri_invariants,
        tags=("petri-net", "invariants", "exact"),
        discovery_terms=("p-invariants", "t-invariants", "place invariants"),
        examples=(
            OperationExample(
                name="producer_consumer_invariants",
                description="Invariants of a one-place producer/consumer net.",
                input={"net": _PRODUCER_CONSUMER_NET},
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.siphon_trap.check",
        title="Check for siphons and traps in a Petri net",
        description="Return all minimal siphons and minimal traps of the given Petri net. "
        "A siphon is a set of places that never gains tokens once it loses "
        "them; a trap is a set of places that never loses tokens once it has "
        "them.",
        request_type=SiphonTrapRequest,
        result_type=SiphonTrapResult,
        run=compute_siphon_trap,
        tags=("petri-net", "siphon", "trap", "exact"),
        examples=(
            OperationExample(
                name="cyclic_net",
                description="Siphons and traps of a cyclic 2-place net.",
                input={
                    "net": {
                        "place_count": 2,
                        "transition_count": 2,
                        "pre": [[1, 0], [0, 1]],
                        "post": [[0, 1], [1, 0]],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.siphon_trap_family.enumerate.compute",
        title="Enumerate all siphons and traps of a Petri net",
        description=(
            "Return every nonempty siphon and trap subset of the net's places. "
            "This complete subset scan is admitted by exact work and output bounds. "
            "Siphon and trap membership use producer/consumer transition supports."
        ),
        request_type=SiphonTrapFamilyRequest,
        result_type=SiphonTrapFamilyResult,
        run=compute_siphon_trap_family,
        tags=("petri-net", "siphon", "trap", "enumeration", "exact"),
        examples=(
            OperationExample(
                name="two_place_cycle",
                description="The full place set is both a siphon and a trap.",
                input={
                    "net": {
                        "place_count": 2,
                        "transition_count": 2,
                        "pre": [[1, 0], [0, 1]],
                        "post": [[0, 1], [1, 0]],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.place_set.support_profile.compute",
        title="Compute a Petri-net place-set support profile",
        description=(
            "For an explicitly selected place subset, return all transitions "
            "producing into it or consuming from it, plus complete offender "
            "sets for the exact siphon and trap predicates. A siphon has every "
            "producer also consume from the subset; a trap has every consumer "
            "also produce into it."
        ),
        request_type=PlaceSetSupportRequest,
        result_type=PlaceSetSupportResult,
        run=compute_place_set_support,
        tags=("petri-net", "siphon", "trap", "support", "exact"),
        discovery_terms=(
            "place set",
            "siphon profile",
            "trap profile",
            "producer consumer support",
        ),
        examples=(
            OperationExample(
                name="producer_consumer_subset",
                description="Check the singleton place in a producer-consumer net.",
                input={
                    "net": _PRODUCER_CONSUMER_NET,
                    "places": {"places": [0]},
                },
            ),
        ),
    ),
    MathTool(
        operation_id="petri_net.place_set.initial_marking_profile.compute",
        title="Profile a Petri-net place set in an initial marking",
        description=(
            "Return the selected places' exact token total, their complete "
            "producer/consumer support profile, and applicable persistence "
            "implications: an initially empty siphon remains empty, and an "
            "initially marked trap remains marked along every firing sequence."
        ),
        request_type=PlaceSetInitialMarkingProfileRequest,
        result_type=PlaceSetInitialMarkingProfileResult,
        run=compute_place_set_initial_marking_profile,
        tags=("petri-net", "siphon", "trap", "marking", "exact"),
        discovery_terms=(
            "initial marking profile",
            "siphon token preservation",
            "trap token preservation",
            "selected token total",
        ),
        examples=(
            OperationExample(
                name="marked_single_place_trap",
                description=(
                    "A self-loop place is both a siphon and trap; its initial "
                    "token is retained by the applicable trap implication."
                ),
                input={
                    "net": {
                        "place_count": 1,
                        "transition_count": 1,
                        "pre": [[1]],
                        "post": [[1]],
                    },
                    "places": {"places": [0]},
                    "marking": {"tokens": [1]},
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
