"""Public native delta-matroid API contract."""

from __future__ import annotations

from jacobian.math.combinatorics.matroids import delta as delta_matroids


def test_public_api_is_small_and_canonical() -> None:
    assert delta_matroids.__all__ == [
        "DeltaMatroidFeasibleSizeProfile",
        "DeltaMatroidTwistWidthProfile",
        "FiniteDeltaMatroid",
        "binary",
        "direct_sum",
        "distance",
        "dual",
        "feasible_size_profile",
        "from_feasible_sets",
        "loop_complement",
        "minor",
        "relabel",
        "twist",
        "twist_width_profile",
        "verify_from_feasible_sets",
        "width",
    ]
