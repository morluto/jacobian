"""Exact rational planar-geometry operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jacobian.math_tools import MathTools

__all__ = ["geometry_operations"]


def geometry_operations() -> MathTools:
    from jacobian.domains.geometry.lines import LINE_OPERATIONS
    from jacobian.domains.geometry.points import POINT_OPERATIONS
    from jacobian.domains.geometry.polygons import POLYGON_OPERATIONS
    from jacobian.domains.geometry.segments import SEGMENT_OPERATIONS
    from jacobian.domains.geometry.triangles import TRIANGLE_OPERATIONS

    return (
        *POINT_OPERATIONS,
        *SEGMENT_OPERATIONS,
        *LINE_OPERATIONS,
        *TRIANGLE_OPERATIONS,
        *POLYGON_OPERATIONS,
    )
