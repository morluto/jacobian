"""Typed contracts for the pinned distance profile operation."""

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.geometry.exact._models import PointConfiguration


class PinnedDistanceProfileRequest(StrictModel):
    """Request the pinned distance support profile."""

    configuration: PointConfiguration


class PinnedDistanceEntry(StrictModel):
    """One source point and its distance partition."""

    source_label: str
    squared_distance: CanonicalRational
    target_labels: tuple[str, ...]


class PinnedDistancePointProfile(StrictModel):
    """Complete distance partition for one source point."""

    source_label: str
    entries: tuple[PinnedDistanceEntry, ...]


class PinnedDistanceProfileResult(StrictModel):
    """Complete pinned distance profile over all source points."""

    configuration: PointConfiguration
    profiles: tuple[PinnedDistancePointProfile, ...]


__all__ = [
    "PinnedDistanceEntry",
    "PinnedDistanceProfileRequest",
    "PinnedDistanceProfileResult",
    "PinnedDistancePointProfile",
]
