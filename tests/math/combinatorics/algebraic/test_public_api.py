"""Exact public API contract for jacobian.math.combinatorics.algebraic."""

from __future__ import annotations

from jacobian.math.combinatorics import algebraic as algebraic_combinatorics


def test_exact_public_api_symbols() -> None:
    """Exact owner-local contract for the algebraic_combinatorics public API."""
    expected = (
        "Biword",
        "BiwordRSKPair",
        "EndpointProfileEntry",
        "EndpointProfileResult",
        "NonnegativeIntegerMatrix",
        "PartitionDominanceResult",
        "RSKTableauPair",
        "SemistandardTableauCheckResult",
        "SemistandardYoungTableauCountResult",
        "StandardTableauCheckResult",
        "WeightedOrderedWord",
        "check_semistandard_tableau",
        "check_standard_tableau",
        "compute_endpoint_profile",
        "conjugate_partition",
        "greene",
        "hook_lengths",
        "inverse_biword",
        "inverse_matrix",
        "inverse_row_insertion_rsk",
        "knuth_moves",
        "matrix_biword",
        "normalize_biword",
        "partition_dominance",
        "row_insertion_rsk",
        "rsk_biword",
        "semistandard_young_tableaux_count",
        "standard_young_tableaux_count",
        "verify_rsk",
    )
    assert tuple(algebraic_combinatorics.__all__) == expected
    assert len(algebraic_combinatorics.__all__) == len(
        set(algebraic_combinatorics.__all__)
    )
    assert all(not name.startswith("_") for name in algebraic_combinatorics.__all__)
    assert all(
        hasattr(algebraic_combinatorics, name)
        for name in algebraic_combinatorics.__all__
    )
