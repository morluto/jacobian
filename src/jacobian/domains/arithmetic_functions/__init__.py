"""Exact arithmetic-function operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["arithmetic_functions_operations"]


def arithmetic_functions_operations() -> MathTools:
    from jacobian.domains.arithmetic_functions.math_tools import (
        ARITHMETIC_FUNCTIONS_OPERATIONS,
    )

    return ARITHMETIC_FUNCTIONS_OPERATIONS
