"""Exact optimization operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["rational_optimization_operations"]


def rational_optimization_operations() -> MathTools:
    from jacobian.domains.optimization.operations import RATIONAL_LINEAR_OPERATIONS

    return RATIONAL_LINEAR_OPERATIONS
