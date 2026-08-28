"""Petri net operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


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
    _op(
        "petri_net.enabled_transitions.compute",
        "Find enabled transitions in a Petri net",
        "Return the indices of all transitions enabled at the given marking, "
        "where a transition is enabled iff every place has enough tokens for "
        "its pre-condition.",
        EnabledTransitionsRequest,
        EnabledTransitionsResult,
        compute_enabled_transitions,
        "petri-net",
        "enabled",
        "exact",
        examples=(
            example(
                "simple_net",
                "Enabled transitions of a 2-place, 2-transition net.",
                {
                    "net": _NET["net"],
                    "marking": {"tokens": [2, 0]},
                },
            ),
        ),
    ),
    _op(
        "petri_net.fire_transition.compute",
        "Fire one transition in a Petri net",
        "Fire a single transition at the given marking and return whether it "
        "succeeded and the resulting marking. If the transition is not "
        "enabled, it does not fire.",
        FireTransitionRequest,
        FireTransitionResult,
        compute_fire_transition,
        "petri-net",
        "firing",
        "exact",
        examples=(
            example(
                "fire_t0",
                "Fire transition 0 from marking [2, 0].",
                {
                    "net": _NET["net"],
                    "marking": {"tokens": [2, 0]},
                    "transition": 0,
                },
            ),
        ),
    ),
    _op(
        "petri_net.incidence_matrix.compute",
        "Compute the incidence matrix of a Petri net",
        "Return the incidence matrix C = Post - Pre for the given Petri net.",
        IncidenceMatrixRequest,
        IncidenceMatrixResult,
        compute_incidence,
        "petri-net",
        "incidence",
        "exact",
        examples=(
            example(
                "simple_net_incidence",
                "Incidence matrix of a 2-place, 2-transition net.",
                {"net": _NET2["net"]},
            ),
        ),
    ),
    _op(
        "petri_net.reachability_graph.compute",
        "Compute the bounded reachability graph of a Petri net",
        "Return the bounded reachability graph from an initial marking via "
        "BFS, including all reachable markings and firing edges. The graph "
        "is truncated at max_states to bound the state space.",
        ReachabilityRequest,
        ReachabilityResult,
        compute_reachability,
        "petri-net",
        "reachability",
        "exact",
        examples=(
            example(
                "simple_net_reachability",
                "Reachability graph of a simple 2-place net.",
                {
                    "net": _NET["net"],
                    "initial_marking": {"tokens": [1, 0]},
                    "max_states": 100,
                },
            ),
        ),
    ),
    _op(
        "petri_net.siphon_trap.check",
        "Check for siphons and traps in a Petri net",
        "Return all minimal siphons and minimal traps of the given Petri net. "
        "A siphon is a set of places that never gains tokens once it loses "
        "them; a trap is a set of places that never loses tokens once it has "
        "them.",
        SiphonTrapRequest,
        SiphonTrapResult,
        compute_siphon_trap,
        "petri-net",
        "siphon",
        "trap",
        "exact",
        examples=(
            example(
                "cyclic_net",
                "Siphons and traps of a cyclic 2-place net.",
                {
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
