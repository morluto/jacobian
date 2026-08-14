"""Exact finite-poset operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["finite_poset_operations"]


def finite_poset_operations() -> MathTools:
    from jacobian.domains.posets.operations import FINITE_POSET_OPERATIONS

    return FINITE_POSET_OPERATIONS
