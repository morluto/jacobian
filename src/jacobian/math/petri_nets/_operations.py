"""Domain adapter for Petri net operations."""

from __future__ import annotations

from jacobian.math.petri_nets._models import (
    EnabledTransitionsRequest,
    EnabledTransitionsResult,
    FireTransitionRequest,
    FireTransitionResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
    ReachabilityRequest,
    ReachabilityResult,
)
from jacobian.math.petri_nets.operations import (
    compute_incidence_matrix,
    enabled_transitions,
    fire_transition,
    reachability_graph,
)

__all__ = [
    "compute_enabled_transitions",
    "compute_fire_transition",
    "compute_incidence",
    "compute_reachability",
]


def compute_enabled_transitions(
    request: EnabledTransitionsRequest,
) -> EnabledTransitionsResult:
    return EnabledTransitionsResult(
        transitions=tuple(enabled_transitions(request.net, request.marking))
    )


def compute_fire_transition(request: FireTransitionRequest) -> FireTransitionResult:
    success, new_marking = fire_transition(
        request.net, request.marking, request.transition
    )
    return FireTransitionResult(fired=success, new_marking=new_marking)


def compute_incidence(request: IncidenceMatrixRequest) -> IncidenceMatrixResult:
    return IncidenceMatrixResult(
        incidence=compute_incidence_matrix(request.net)
    )


def compute_reachability(request: ReachabilityRequest) -> ReachabilityResult:
    states, edges, truncated = reachability_graph(
        request.net, request.initial_marking, request.max_states
    )
    return ReachabilityResult(
        states=tuple(states),
        edges=tuple(edges),
        truncated=truncated,
    )
