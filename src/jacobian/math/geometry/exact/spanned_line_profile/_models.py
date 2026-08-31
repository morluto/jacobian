"""Typed contracts for the spanned-line profile operation."""

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.geometry.exact._models import PointConfiguration


def _require_coordinate_distinctness(configuration: PointConfiguration) -> None:
    coordinates = [
        tuple((coordinate.num, coordinate.den) for coordinate in point.coordinates)
        for point in configuration.points
    ]
    if len(coordinates) != len(set(coordinates)):
        raise PydanticCustomError(
            "geometry.spanned_line_profile.points_distinct",
            "all points must have pairwise distinct coordinates",
        )


class SpannedLineProfileRequest(StrictModel):
    """Request the pair-spanned affine line profile."""

    configuration: PointConfiguration

    @model_validator(mode="after")
    def require_distinct_points(self) -> "SpannedLineProfileRequest":
        _require_coordinate_distinctness(self.configuration)
        return self


class SpannedLineEntry(StrictModel):
    """One affine line spanned by source pairs(s)."""

    source_pairs: tuple[tuple[int, int], ...]
    point_count: int


class SpannedLineProfileResult(StrictModel):
    """Complete pair-spanned affine line profile."""

    configuration: PointConfiguration
    lines: tuple[SpannedLineEntry, ...]
    line_count: int


__all__ = [
    "SpannedLineEntry",
    "SpannedLineProfileRequest",
    "SpannedLineProfileResult",
]
