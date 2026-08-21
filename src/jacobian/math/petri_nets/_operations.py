"""Domain adapter for Petri net operations."""

from __future__ import annotations

from jacobian.math.petri_nets._models import (
    EnabledTransitionsRequest,
    EnabledTransitionsResult,
    FireTransitionRequest,
    FireTransitionResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
    ReachabilityEnvelopeEscape,
    ReachabilityFrontier,
    ReachabilityRequest,
    ReachabilityResult,
)
from jacobian.math.petri_nets.operations import (
    compute_incidence_matrix,
    enabled_transitions,
    fire_transition,
    reachability_graph,
)
from jacobian.math.petri_nets.values import MAX_PETRI_MARKING, Marking

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
    states, edges, frontier, envelope_escape = reachability_graph(
        request.net, request.initial_marking, request.max_states
    )
    return ReachabilityResult(
        net=request.net,
        initial_marking=request.initial_marking,
        max_states=request.max_states,
        states=tuple(states),
        edges=tuple(edges),
        status=(
            "ESCAPES_DECLARED_ENVELOPE"
            if envelope_escape is not None
            else "TRUNCATED"
            if frontier
            else "COMPLETE"
        ),
        frontier=tuple(
            ReachabilityFrontier(
                source_state=source,
                transition=transition,
                target_marking=target,
            )
            for source, transition, target in frontier
        ),
        envelope_escape=(
            None
            if envelope_escape is None
            else ReachabilityEnvelopeEscape(
                source_state=envelope_escape[0],
                transition=envelope_escape[1],
                target_marking=envelope_escape[2],
            )
        ),
    )
