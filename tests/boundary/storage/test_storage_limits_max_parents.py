"""Focused tests for the StorageLimits max_parents construction invariant.

The runtime's search archive pages without witness lineage have three fixed
parents and require at least one candidate lineage parent. ``max_parents >=
4`` is therefore enforced at construction, rather than allowing a
configuration that every such search start would reject later.
"""

from __future__ import annotations

import pytest

from jacobian.storage.models import StorageLimits


def test_default_storage_limits_keeps_max_parents() -> None:
    limits = StorageLimits()
    assert limits.max_parents == 4096


def test_storage_limits_accepts_minimum_search_capacity() -> None:
    assert StorageLimits(max_parents=4).max_parents == 4


@pytest.mark.parametrize("value", [0, 1, 2, 3, -1])
def test_storage_limits_rejects_below_minimum_search_capacity(value: int) -> None:
    with pytest.raises(ValueError, match="max_parents must be at least 4"):
        StorageLimits(max_parents=value)
