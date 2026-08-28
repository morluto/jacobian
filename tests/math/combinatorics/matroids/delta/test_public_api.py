"""Public native delta-matroid API contract."""

from __future__ import annotations

from jacobian.math.combinatorics.matroids import delta as delta_matroids


def test_public_api_is_small_and_canonical() -> None:
    assert delta_matroids.__all__ == ["FiniteDeltaMatroid", "from_feasible_sets"]
