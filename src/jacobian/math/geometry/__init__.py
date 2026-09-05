"""Exact geometry operation ownership."""

from jacobian.math.geometry._convex_polygon_intersection import (
    ConvexPolygonIntersectionResult,
    ConvexRationalPolygon,
)
from jacobian.math.geometry.operations import convex_polygon_intersection

__all__ = [
    "ConvexPolygonIntersectionResult",
    "ConvexRationalPolygon",
    "convex_polygon_intersection",
]
