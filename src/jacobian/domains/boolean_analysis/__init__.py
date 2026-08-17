"""Exact Boolean function analysis operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["boolean_analysis_operations"]


def boolean_analysis_operations() -> MathTools:
    from jacobian.domains.boolean_analysis.math_tools import (
        BOOLEAN_ANALYSIS_OPERATIONS,
    )

    return BOOLEAN_ANALYSIS_OPERATIONS
