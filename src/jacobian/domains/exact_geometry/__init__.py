"""Exact geometry operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["exact_geometry_operations"]


def exact_geometry_operations() -> MathTools:
    from jacobian.domains.exact_geometry.math_tools import (
        EXACT_GEOMETRY_OPERATIONS,
    )

    return EXACT_GEOMETRY_OPERATIONS
