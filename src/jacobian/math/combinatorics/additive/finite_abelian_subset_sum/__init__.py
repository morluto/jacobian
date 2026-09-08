"""Complete indexed subset-sum profiles in finite abelian product groups."""

from __future__ import annotations

from typing import Any

__all__ = [
    "finite_abelian_subset_sum_profile",
]


def __getattr__(name: str) -> Any:
    if name == "finite_abelian_subset_sum_profile":
        from jacobian.math.combinatorics.additive.finite_abelian_subset_sum.operations import (
            finite_abelian_subset_sum_profile,
        )

        return finite_abelian_subset_sum_profile
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
