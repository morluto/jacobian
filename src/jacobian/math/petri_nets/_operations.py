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
    SiphonTrapRequest,
    SiphonTrapResult,
)
from jacobian.math.petri_nets.operations import (
    compute_incidence_matrix,
    enabled_transitions,
    find_minimal_siphons,
    find_minimal_traps,
    fire_transition,
    reachability_graph,
)
from jacobian.math.petri_nets.values import MAX_PETRI_MARKING, Marking

__all__ = [
    "compute_enabled_transitions",
    "compute_fire_transition",
    "compute_incidence",
    "compute_reachability",
    "compute_siphon_trap",
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
    if any(token > MAX_PETRI_MARKING for token in new_marking):
        return FireTransitionResult(
            status="ESCAPES_DECLARED_ENVELOPE",
            envelope_escape=new_marking,
        )
    return FireTransitionResult(
        status="FIRED" if success else "NOT_ENABLED",
        new_marking=Marking(tokens=new_marking),
    )


def compute_incidence(request: IncidenceMatrixRequest) -> IncidenceMatrixResult:
    return IncidenceMatrixResult(incidence=compute_incidence_matrix(request.net))


def compute_reachability(request: ReachabilityRequest) -> ReachabilityResult:
    states, edges, truncated = reachability_graph(
        request.net, request.initial_marking, request.max_states
    )
    return ReachabilityResult(
        states=tuple(states),
        edges=tuple(edges),
        truncated=truncated,
    )


def compute_siphon_trap(request: SiphonTrapRequest) -> SiphonTrapResult:
    siphons = find_minimal_siphons(request.net)
    traps = find_minimal_traps(request.net)
    return SiphonTrapResult(
        siphons=tuple(tuple(sorted(s)) for s in siphons),
        traps=tuple(tuple(sorted(t)) for t in traps),
    )
