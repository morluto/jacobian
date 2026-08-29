"""Typed contracts for the triangle area profile operation."""

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.geometry.exact._models import PointConfiguration


class TriangleAreaProfileRequest(StrictModel):
    """Request the triangle area profile of a planar configuration."""

    configuration: PointConfiguration


class TriangleAreaEntry(StrictModel):
    """One triangle and its exact unsigned area."""

    indices: tuple[int, int, int]
    area: CanonicalRational


class TriangleAreaProfileResult(StrictModel):
    """Complete triangle-area profile of a planar point configuration."""

    configuration: PointConfiguration
    entries: tuple[TriangleAreaEntry, ...]
    area_classes: tuple[tuple[CanonicalRational, tuple[tuple[int, int, int], ...]], ...]


__all__ = [
    "TriangleAreaEntry",
    "TriangleAreaProfileRequest",
    "TriangleAreaProfileResult",
]
