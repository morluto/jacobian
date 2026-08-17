"""Exact rational planar-geometry operations."""

from jacobian.catalog.models import MathTools
from jacobian.math.geometry._lines import LINE_OPERATIONS
from jacobian.math.geometry._points import POINT_OPERATIONS
from jacobian.math.geometry._polygons import POLYGON_OPERATIONS
from jacobian.math.geometry._segments import SEGMENT_OPERATIONS
from jacobian.math.geometry._triangles import TRIANGLE_OPERATIONS

__all__ = ["TOOLS"]

TOOLS: MathTools = (
    *POINT_OPERATIONS,
    *SEGMENT_OPERATIONS,
    *LINE_OPERATIONS,
    *TRIANGLE_OPERATIONS,
    *POLYGON_OPERATIONS,
)
