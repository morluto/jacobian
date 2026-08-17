"""Exact algebraic combinatorics operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["algebraic_combinatorics_operations"]


def algebraic_combinatorics_operations() -> MathTools:
    from jacobian.domains.algebraic_combinatorics.math_tools import (
        ALGEBRAIC_COMBINATORICS_OPERATIONS,
    )

    return ALGEBRAIC_COMBINATORICS_OPERATIONS
