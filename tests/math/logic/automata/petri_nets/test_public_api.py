"""Exact public API contract for jacobian.math.logic.automata.petri_nets."""

from __future__ import annotations

from jacobian.math.logic.automata import petri_nets


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the petri_nets public API."""
    expected = (
        "Marking",
        "PetriNet",
        "compute_incidence_matrix",
        "enabled_transitions",
        "find_minimal_siphons",
        "find_minimal_traps",
        "fire_transition",
        "reachability_graph",
    )
    assert tuple(petri_nets.__all__) == expected
    assert len(petri_nets.__all__) == len(set(petri_nets.__all__))
    assert all(not name.startswith("_") for name in petri_nets.__all__)
    assert all(hasattr(petri_nets, name) for name in petri_nets.__all__)
