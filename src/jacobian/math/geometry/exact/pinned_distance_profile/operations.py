"""Pinned distance profile kernel."""

from __future__ import annotations

from fractions import Fraction
from itertools import groupby

from jacobian._exact import CanonicalRational
from jacobian.math.geometry.exact._models import PointConfiguration
from jacobian.math.geometry.exact.pinned_distance_profile._models import (
    PinnedDistanceEntry,
    PinnedDistancePointProfile,
    PinnedDistanceProfileResult,
)

__all__ = ["compute_pinned_distance_profile"]


def compute_pinned_distance_profile(
    configuration: PointConfiguration,
) -> PinnedDistanceProfileResult:
    """Return for every source point its complete sorted distance partition.

    For each point i, partition all other points j by the exact squared
    Euclidean distance d(i,j), with labels sorted within each distance class.
    """
    points = configuration.points
    n = len(points)

    profiles: list[PinnedDistancePointProfile] = []

    for i in range(n):
        source_label = points[i].label
        coords_i = [c.as_fraction() for c in points[i].coordinates]

        dist_labels: list[tuple[Fraction, str]] = []
        for j in range(n):
            if i == j:
                continue
            coords_j = [c.as_fraction() for c in points[j].coordinates]
            sq_dist = sum(
                (coords_i[d] - coords_j[d]) ** 2
                for d in range(len(coords_i))
            )
            dist_labels.append((sq_dist, points[j].label))

        dist_labels.sort(key=lambda x: (x[0], x[1]))

        entries: list[PinnedDistanceEntry] = []
        for dist, group in groupby(dist_labels, key=lambda x: x[0]):
            labels = tuple(sorted(l for _, l in group))
            entries.append(
                PinnedDistanceEntry(
                    source_label=source_label,
                    squared_distance=CanonicalRational.from_fraction(dist),
                    target_labels=labels,
                )
            )

        profiles.append(
            PinnedDistancePointProfile(
                source_label=source_label,
                entries=tuple(entries),
            )
        )

    return PinnedDistanceProfileResult(
        configuration=configuration,
        profiles=tuple(profiles),
    )
