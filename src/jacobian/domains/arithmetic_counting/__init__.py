"""Exact arithmetic counting operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["arithmetic_counting_operations"]


def arithmetic_counting_operations() -> MathTools:
    from jacobian.domains.arithmetic_counting.math_tools import (
        ARITHMETIC_COUNTING_OPERATIONS,
    )

    return ARITHMETIC_COUNTING_OPERATIONS
