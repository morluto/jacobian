"""Typed contracts for the triangle area profile operation."""

from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.geometry.exact._models import PointConfiguration


def _require_distinct_coordinates(configuration: PointConfiguration) -> None:
    coordinate_keys = [
        tuple((coordinate.num, coordinate.den) for coordinate in point.coordinates)
        for point in configuration.points
    ]
    if len(coordinate_keys) != len(set(coordinate_keys)):
        raise PydanticCustomError(
            "geometry.triangle_area_points_distinct",
            "triangle area profiles require pairwise distinct point coordinates",
        )


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
