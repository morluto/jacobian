"""Focused tests for the StorageLimits max_parents construction invariant.

PR3 requires ``max_parents >= 3`` at construction so a misconfigured store
cannot fragment the parent-archive chain below the search service's fixed-page
assumption.  The check runs in ``__post_init__`` and refuses construction, not
clamping or warning.
"""

from __future__ import annotations

import pytest

from jacobian.storage.models import StorageLimits


def test_default_storage_limits_keeps_max_parents() -> None:
    limits = StorageLimits()
    assert limits.max_parents == 4096


def test_storage_limits_accepts_three_parents() -> None:
    assert StorageLimits(max_parents=3).max_parents == 3


@pytest.mark.parametrize("value", [0, 1, 2, -1])
def test_storage_limits_rejects_below_three(value: int) -> None:
    with pytest.raises(ValueError, match="max_parents must be at least 3"):
        StorageLimits(max_parents=value)
