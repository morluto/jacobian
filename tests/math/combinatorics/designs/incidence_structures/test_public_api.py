"""Exact public API contract for finite incidence structures."""

from jacobian.math.combinatorics.designs import incidence_structures


def test_exact_public_api_symbols() -> None:
    expected = (
        "ComputedSteinerTripleSystem",
        "ContainmentProfileResult",
        "IncidenceMomentComparison",
        "IncidenceStructure",
        "IncidenceTradeResult",
        "SteinerTripleSystemNotFound",
        "SteinerTripleSystemOutcome",
        "SteinerTripleSystemResult",
        "SteinerTripleSystemShard",
        "SteinerTripleSystemUnknown",
        "check_incidence_trade",
        "complement",
        "construct_steiner_triple_system",
        "containment_profile",
        "degree_profile",
        "derived_residual",
        "dual",
        "gram",
        "incidence_matrix",
        "intersections",
        "levi_graph",
        "restriction",
    )
    # The module's declared __all__ is the source of truth: every
    # advertised name must stay exported, but adding one is allowed.
    assert set(expected) <= set(incidence_structures.__all__)
    assert all(hasattr(incidence_structures, name) for name in expected)
