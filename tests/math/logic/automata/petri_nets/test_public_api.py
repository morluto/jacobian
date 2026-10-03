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
        "disjoint_union",
        "enabled_transitions",
        "find_minimal_siphons",
        "find_minimal_traps",
        "fire_transition",
        "marking_commutation_profile",
        "marking_conflict_profile",
        "marking_equation",
        "marking_reachability",
        "petri_invariants",
        "petri_net_matrices",
        "place_set_initial_marking_profile",
        "place_set_support",
        "reachability_graph",
        "reachability_terminal_scc_profile",
        "reachable_dead_markings",
        "relabel_petri_net",
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
    # The module's declared __all__ is the source of truth: every
    # advertised name must stay exported, but adding one is allowed.
    assert set(expected) <= set(petri_nets.__all__)
    assert len(petri_nets.__all__) == len(set(petri_nets.__all__))
    assert all(not name.startswith("_") for name in petri_nets.__all__)
    assert all(hasattr(petri_nets, name) for name in petri_nets.__all__)
