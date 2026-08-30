"""Typed contracts for the pinned distance support profile operation."""

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.geometry.exact._models import PointConfiguration


class PinnedDistanceSupportProfileRequest(StrictModel):
    """Request for the pinned distance support profile."""

    configuration: PointConfiguration


class DistanceClass(StrictModel):
    """One distance class at a source point."""

    squared_distance: CanonicalRational
    target_labels: tuple[str, ...]


class PinnedDistanceEntry(StrictModel):
    """One source point's distance partition."""

    source_label: str
    distance_classes: tuple[DistanceClass, ...]


class PinnedDistanceSupportProfileResult(StrictModel):
    """The complete pinned distance support profile."""

    configuration: PointConfiguration
    entries: tuple[PinnedDistanceEntry, ...]


__all__ = [
    "DistanceClass",
    "PinnedDistanceEntry",
    "PinnedDistanceSupportProfileRequest",
    "PinnedDistanceSupportProfileResult",
]
