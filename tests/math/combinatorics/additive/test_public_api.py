"""Exact public API contract for ``jacobian.math.combinatorics.additive``."""

from __future__ import annotations

from jacobian.math.combinatorics import additive as additive_combinatorics


def test_exact_public_api_symbols() -> None:
    """Keep canonical additive-combinatorics values and kernels explicit."""

    expected = (
        "IndexSubset",
        "IndexedIntegerSequence",
        "SubsetSumProfile",
        "SubsetSumProfileEntry",
        "subset_sum_profile",
    )
    assert tuple(additive_combinatorics.__all__) == expected
    assert len(additive_combinatorics.__all__) == len(
        set(additive_combinatorics.__all__)
    )
    assert all(not name.startswith("_") for name in additive_combinatorics.__all__)
    assert all(
        hasattr(additive_combinatorics, name) for name in additive_combinatorics.__all__
    )
