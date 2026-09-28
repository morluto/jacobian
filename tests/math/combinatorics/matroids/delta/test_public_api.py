"""Public native delta-matroid API contract."""

from __future__ import annotations

from jacobian.math.combinatorics.matroids import delta as delta_matroids


def test_public_api_is_small_and_canonical() -> None:
    assert delta_matroids.__all__ == [
        "DeltaMatroidDistanceProfile",
        "DeltaMatroidFeasibleSizeProfile",
        "DeltaMatroidTwistWidthProfile",
        "DistanceInterlaceResult",
        "FiniteDeltaMatroid",
        "binary",
        "binary_matrix_twist",
        "direct_sum",
        "distance",
        "distance_interlace_polynomial",
        "distance_profile",
        "dual",
        "feasible_size_profile",
        "from_feasible_sets",
        "loop_complement",
        "lower_matroid",
        "minor",
        "relabel",
        "twist",
        "twist_polynomial",
        "twist_width_profile",
        "upper_matroid",
        "verify_from_feasible_sets",
        "width",
    ]
