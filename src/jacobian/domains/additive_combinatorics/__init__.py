"""Exact additive combinatorics operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["additive_combinatorics_operations"]


def additive_combinatorics_operations() -> MathTools:
    from jacobian.domains.additive_combinatorics.math_tools import (
        ADDITIVE_COMBINATORICS_OPERATIONS,
    )

    return ADDITIVE_COMBINATORICS_OPERATIONS
