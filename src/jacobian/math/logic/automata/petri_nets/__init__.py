"""Supported native Petri-net API."""

from jacobian.math.logic.automata.petri_nets.operations import (
    compute_incidence_matrix,
    enabled_transitions,
    find_minimal_siphons,
    find_minimal_traps,
    fire_transition,
    petri_invariants,
    reachability_graph,
    replay_firing_sequence,
    verify_enabled_transitions,
    verify_fire_transition,
    verify_firing_sequence_replay,
    verify_incidence_matrix,
    verify_invariants,
    verify_reachability_graph,
    verify_siphon_trap,
)
from jacobian.math.logic.automata.petri_nets.values import Marking, PetriNet

__all__ = [
    "Marking",
    "PetriNet",
    "compute_incidence_matrix",
    "enabled_transitions",
    "find_minimal_siphons",
    "find_minimal_traps",
    "fire_transition",
    "petri_invariants",
    "reachability_graph",
    "replay_firing_sequence",
    "verify_enabled_transitions",
    "verify_fire_transition",
    "verify_firing_sequence_replay",
    "verify_incidence_matrix",
    "verify_invariants",
    "verify_reachability_graph",
    "verify_siphon_trap",
]
