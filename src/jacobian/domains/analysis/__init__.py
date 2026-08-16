"""Validated real-analysis operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["real_analysis_operations"]


def real_analysis_operations() -> MathTools:
    from jacobian.domains.analysis.operations import ANALYSIS_OPERATIONS

    return ANALYSIS_OPERATIONS
