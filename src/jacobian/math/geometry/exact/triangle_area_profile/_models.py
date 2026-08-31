"""Typed contracts for the triangle area profile operation."""

from typing import Self

from pydantic import model_validator
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

    @model_validator(mode="after")
    def require_planar_configuration(self) -> Self:
        _require_distinct_coordinates(self.configuration)
        if len(self.configuration.points[0].coordinates) != 2:
            raise PydanticCustomError(
                "geometry.triangle_area_planar_configuration",
                "triangle area profiles require exactly two coordinates per point",
            )
        return self


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
