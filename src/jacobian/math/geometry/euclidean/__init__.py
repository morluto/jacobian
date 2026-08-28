"""Exact Euclidean-geometry values and operations."""

from jacobian.math.geometry.euclidean._models import Triangle
from jacobian.math.geometry.euclidean.operations import (
    angles_equal,
    squared_segment_ratio,
    triangles_similar,
)

__all__ = ["Triangle", "angles_equal", "squared_segment_ratio", "triangles_similar"]
