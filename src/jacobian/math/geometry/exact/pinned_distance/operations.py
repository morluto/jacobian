"""Pinned distance support profile kernel."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.exact._models import PointConfiguration
from jacobian.math.geometry.exact.pinned_distance._models import (
    DistanceClass,
    PinnedDistanceEntry,
    PinnedDistanceSupportProfileResult,
)

__all__ = ["compute_pinned_distance_support_profile"]


def compute_pinned_distance_support_profile(
    configuration: PointConfiguration,
) -> PinnedDistanceSupportProfileResult:
    """Return the per-point partition of all other points by squared distance.

    For each source point, group all other points by their exact squared
    Euclidean distance, sorted by increasing distance.
    """
    points = configuration.points
    entries: list[PinnedDistanceEntry] = []

    for i, source in enumerate(points):
        dist_to_targets: dict[Fraction, list[str]] = {}
        for j, target in enumerate(points):
            if i == j:
                continue
            sq_dist = _squared_distance(source.coordinates, target.coordinates)
            dist_to_targets.setdefault(sq_dist, []).append(target.label)

        classes: list[DistanceClass] = []
        for dist in sorted(dist_to_targets):
            classes.append(
                DistanceClass(
                    squared_distance=CanonicalRational.from_fraction(dist),
                    target_labels=tuple(sorted(dist_to_targets[dist])),
                )
            )
        entries.append(
            PinnedDistanceEntry(
                source_label=source.label,
                distance_classes=tuple(classes),
            )
        )

    return PinnedDistanceSupportProfileResult(
        configuration=configuration,
        entries=tuple(entries),
    )


def _squared_distance(
    coords_a: tuple[CanonicalRational, ...],
    coords_b: tuple[CanonicalRational, ...],
) -> Fraction:
    """Compute the exact squared Euclidean distance."""
    total = Fraction(0)
    for a, b in zip(coords_a, coords_b, strict=True):
        diff = a.as_fraction() - b.as_fraction()
        total += diff * diff
    return total
