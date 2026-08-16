"""Exact Euclidean geometry operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["euclidean_geometry_operations"]


def euclidean_geometry_operations() -> MathTools:
    from jacobian.domains.euclidean_geometry.math_tools import (
        EUCLIDEAN_GEOMETRY_OPERATIONS,
    )

    return EUCLIDEAN_GEOMETRY_OPERATIONS
