"""Petri net operation declarations."""

from typing import Any

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.logic.automata.petri_nets._models import (
    MAX_SIPHON_TRAP_PLACES,
    EnabledTransitionsRequest,
    EnabledTransitionsResult,
    FireTransitionRequest,
    FireTransitionResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
    ReachabilityRequest,
    ReachabilityResult,
    SiphonTrapRequest,
    SiphonTrapResult,
)
from jacobian.math.logic.automata.petri_nets.operations import (
    compute_incidence_matrix,
    enabled_transitions,
    fire_transition,
    reachability_graph,
    siphon_trap,
)


def compute_enabled_transitions(
    request: EnabledTransitionsRequest,
) -> EnabledTransitionsResult:
    return enabled_transitions(request.net, request.marking)


def compute_fire_transition(request: FireTransitionRequest) -> FireTransitionResult:
    return fire_transition(request.net, request.marking, request.transition)


def compute_incidence(request: IncidenceMatrixRequest) -> IncidenceMatrixResult:
    return compute_incidence_matrix(request.net)


def compute_reachability(request: ReachabilityRequest) -> ReachabilityResult:
    try:
        return reachability_graph(
            request.net, request.initial_marking, request.max_states
        )
    except ValueError as error:
        raise OperationDomainValidationError(
            location=("net", "max_states"),
            code="petri_net.reachability_bound",
            message=str(error),
        ) from error


def compute_siphon_trap(request: SiphonTrapRequest) -> SiphonTrapResult:
    try:
        return siphon_trap(request.net)
    except ValueError as error:
        code = (
            "petri_net.siphon_trap_place_bound"
            if request.net.place_count > MAX_SIPHON_TRAP_PLACES
            else "petri_net.siphon_trap_work_bound"
        )
        raise OperationDomainValidationError(
            location=("net",),
            code=code,
            message=str(error),
        ) from error


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

TOOLS: tuple[MathTool[Any, Any], ...] = (
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
)


__all__ = ["TOOLS"]
