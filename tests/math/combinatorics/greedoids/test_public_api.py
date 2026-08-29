"""Exact public API contract for jacobian.math.combinatorics.greedoids."""

from __future__ import annotations

from jacobian.math.combinatorics import greedoids


def test_exact_public_api_symbols() -> None:
    expected = (
        "FiniteFeasibleSetSystem",
        "antimatroid_to_convex_geometry",
        "bases",
        "bases_profile",
        "basic_word_outcome",
        "basic_word_profile",
        "convex_geometry_profile",
        "convex_geometry_to_antimatroid",
        "feasible_continuations",
        "rank",
        "rank_profile",
        "recognize",
        "union_closed",
    )
    assert tuple(greedoids.__all__) == expected
    assert len(greedoids.__all__) == len(set(greedoids.__all__))
    assert all(not name.startswith("_") for name in greedoids.__all__)
    assert all(hasattr(greedoids, name) for name in greedoids.__all__)
