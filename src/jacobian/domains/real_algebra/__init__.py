"""Exact real algebra operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["real_algebra_operations"]


def real_algebra_operations() -> MathTools:
    from jacobian.domains.real_algebra.math_tools import REAL_ALGEBRA_OPERATIONS

    return REAL_ALGEBRA_OPERATIONS
