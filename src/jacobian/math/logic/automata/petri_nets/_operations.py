"""Domain adapter for Petri net operations."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.automata.petri_nets._models import (
    MAX_SIPHON_TRAP_PLACES,
    MAX_SIPHON_TRAP_WORK,
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
    find_minimal_siphons,
    find_minimal_traps,
    fire_transition,
    reachability_graph,
)

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
    if request.net.place_count > MAX_SIPHON_TRAP_PLACES:
        raise OperationDomainValidationError(
            location=("net",),
            code="petri_net.siphon_trap_place_bound",
            message=(
                "siphon/trap check supports at most "
                f"{MAX_SIPHON_TRAP_PLACES} places for exact enumeration"
            ),
        )
    candidates = (1 << request.net.place_count) - 1
    work = 2 * candidates * (request.net.transition_count + request.net.place_count)
    if work > MAX_SIPHON_TRAP_WORK:
        raise OperationDomainValidationError(
            location=("net",),
            code="petri_net.siphon_trap_work_bound",
            message=(
                "siphon/trap candidate and transition-scan work exceeds the admitted bound"
            ),
        )
    siphons = find_minimal_siphons(request.net)
    traps = find_minimal_traps(request.net)
    return SiphonTrapResult(
        siphons=tuple(tuple(sorted(s)) for s in siphons),
        traps=tuple(tuple(sorted(t)) for t in traps),
    )
