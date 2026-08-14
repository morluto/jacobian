"""Exact integer and rational arithmetic operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["arithmetic_operations"]


def arithmetic_operations() -> MathTools:
    from jacobian.domains.arithmetic.integers import INTEGER_OPERATIONS
    from jacobian.domains.arithmetic.rationals import RATIONAL_OPERATIONS
    from jacobian.domains.arithmetic.real_quadratic import REAL_QUADRATIC_OPERATIONS

    return (*INTEGER_OPERATIONS, *RATIONAL_OPERATIONS, *REAL_QUADRATIC_OPERATIONS)
