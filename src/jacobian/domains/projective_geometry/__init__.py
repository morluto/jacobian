"""Exact rational projective-geometry operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["projective_geometry_operations"]


def projective_geometry_operations() -> MathTools:
    from jacobian.domains.projective_geometry.arrangements import (
        PROJECTIVE_LINE_ARRANGEMENT_OPERATION,
    )

    return (PROJECTIVE_LINE_ARRANGEMENT_OPERATION,)
