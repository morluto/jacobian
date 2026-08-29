"""Exact public API contract for finite incidence structures."""

from jacobian.math.combinatorics.designs import incidence_structures


def test_exact_public_api_symbols() -> None:
    expected = (
        "ContainmentProfileResult",
        "IncidenceMomentComparison",
        "IncidenceStructure",
        "IncidenceTradeResult",
        "check_incidence_trade",
        "complement",
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
    assert tuple(incidence_structures.__all__) == expected
    assert all(hasattr(incidence_structures, name) for name in expected)
