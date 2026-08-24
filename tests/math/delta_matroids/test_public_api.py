"""Public native delta-matroid API contract."""

from __future__ import annotations


def test_public_api_is_small_and_canonical() -> None:
    from jacobian.math import delta_matroids

    assert delta_matroids.__all__ == ["FiniteDeltaMatroid", "from_feasible_sets"]
