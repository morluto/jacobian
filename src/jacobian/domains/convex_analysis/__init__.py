"""Exact convex analysis operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["convex_analysis_operations"]


def convex_analysis_operations() -> MathTools:
    from jacobian.domains.convex_analysis.math_tools import (
        CONVEX_ANALYSIS_OPERATIONS,
    )

    return CONVEX_ANALYSIS_OPERATIONS
