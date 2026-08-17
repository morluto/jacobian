"""Exact Diophantine approximation operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["diophantine_approximation_operations"]


def diophantine_approximation_operations() -> MathTools:
    from jacobian.domains.diophantine_approximation.math_tools import (
        DIOPHANTINE_APPROXIMATION_OPERATIONS,
    )

    return DIOPHANTINE_APPROXIMATION_OPERATIONS
