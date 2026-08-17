"""Exact discrepancy theory operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["discrepancy_theory_operations"]


def discrepancy_theory_operations() -> MathTools:
    from jacobian.domains.discrepancy_theory.math_tools import (
        DISCREPANCY_THEORY_OPERATIONS,
    )

    return DISCREPANCY_THEORY_OPERATIONS
