"""Public native delta-matroid API contract."""

from __future__ import annotations

from jacobian.math.combinatorics.matroids import delta as delta_matroids


def test_public_api_is_small_and_canonical() -> None:
    assert delta_matroids.__all__ == [
        "DistanceInterlaceResult",
        "FiniteDeltaMatroid",
        "binary",
        "distance_interlace_polynomial",
        "dual",
        "from_feasible_sets",
        "minor",
        "relabel",
        "twist",
        "verify_from_feasible_sets",
        "width",
    ]
