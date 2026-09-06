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
        "additive_energy",
        "direct_sum_predicate",
        "representation_profile",
        "subset_sum_profile",
        "sumset_cardinality",
        "verify_additive_energy",
        "verify_direct_sum_predicate",
        "verify_representation_profile",
        "verify_sumset_cardinality",
    )
    assert tuple(additive_combinatorics.__all__) == expected
    assert len(additive_combinatorics.__all__) == len(
        set(additive_combinatorics.__all__)
    )
    assert all(not name.startswith("_") for name in additive_combinatorics.__all__)
    assert all(
        hasattr(additive_combinatorics, name) for name in additive_combinatorics.__all__
    )
