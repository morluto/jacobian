"""Typed contracts for the spanned-line profile operation."""

from jacobian._models import StrictModel
from jacobian.math.geometry.exact._models import PointConfiguration


class SpannedLineProfileRequest(StrictModel):
    """Request the pair-spanned affine line profile."""

    configuration: PointConfiguration


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
