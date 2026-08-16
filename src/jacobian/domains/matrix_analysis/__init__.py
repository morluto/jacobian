"""Exact matrix analysis operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["matrix_analysis_operations"]


def matrix_analysis_operations() -> MathTools:
    from jacobian.domains.matrix_analysis.math_tools import (
        MATRIX_ANALYSIS_OPERATIONS,
    )

    return MATRIX_ANALYSIS_OPERATIONS
