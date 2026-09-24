"""Exact public API contract for jacobian.math.logic.automata.petri_nets."""

from __future__ import annotations

from jacobian.math.logic.automata import petri_nets


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the petri_nets public API."""
    expected = (
        "Marking",
        "PetriNet",
        "PetriPlaceSubset",
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
        "place_set_initial_marking_profile",
        "place_set_support",
        "reachability_graph",
        "replay_firing_sequence",
        "reverse_petri_net",
        "siphon_trap_family",
        "state_equation_target",
        "verify_enabled_transitions",
        "verify_fire_transition",
        "verify_incidence_matrix",
        "verify_invariants",
        "verify_reachability_graph",
        "verify_siphon_trap",
    )
    assert tuple(petri_nets.__all__) == expected
    assert len(petri_nets.__all__) == len(set(petri_nets.__all__))
    assert all(not name.startswith("_") for name in petri_nets.__all__)
    assert all(hasattr(petri_nets, name) for name in petri_nets.__all__)
