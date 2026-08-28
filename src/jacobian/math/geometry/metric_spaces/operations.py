"""Exact finite metric space kernels."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational

from ._models import (
    BallResult,
    EccentricityResult,
    FiniteMetricSpace,
    GromovHyperbolicityResult,
    MetricProfileResult,
)

__all__ = ["ball", "gromov_hyperbolicity", "metric_profile"]


def metric_profile(
    metric_space: FiniteMetricSpace,
) -> MetricProfileResult:
    """Compute diameter, radius, eccentricities, centers, and periphery."""
    distances = metric_space.distances
    n = len(distances)
    eccentricities = [max(distances[i]) for i in range(n)]
    diameter = max(eccentricities)
    radius = min(eccentricities)
    centers = tuple(i for i, e in enumerate(eccentricities) if e == radius)
    periphery = tuple(i for i, e in enumerate(eccentricities) if e == diameter)
    return MetricProfileResult(
        diameter=diameter,
        radius=radius,
        eccentricities=tuple(
            EccentricityResult(point=index, eccentricity=eccentricity)
            for index, eccentricity in enumerate(eccentricities)
        ),
        centers=centers,
        periphery=periphery,
    )


def ball(metric_space: FiniteMetricSpace, center: int, radius: int) -> BallResult:
    """Return the list of points within radius of center."""
    if not 0 <= center < metric_space.point_count:
        raise ValueError("center index must be within the metric space")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    distances = metric_space.distances
    n = len(distances)
    return BallResult(
        center=center,
        radius=radius,
        points=tuple(i for i in range(n) if distances[center][i] <= radius),
    )


def gromov_hyperbolicity(
    metric_space: FiniteMetricSpace,
) -> GromovHyperbolicityResult:
    """Compute the four-point Gromov hyperbolicity (max over all quadruples).

    For four points i, j, k, l, define the three pairing sums
    s1 = d(i,j)+d(k,l), s2 = d(i,k)+d(j,l), s3 = d(i,l)+d(j,k).
    The four-point delta is half the gap between the two largest sums, and
    the hyperbolicity is the maximum delta over all quadruples. Since that
    gap can be odd, the result is an exact ``Fraction`` (possibly half-integer).
    """
    distances = metric_space.distances
    n = len(distances)
    max_delta = Fraction(0)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for m in range(k + 1, n):
                    s1 = distances[i][j] + distances[k][m]
                    s2 = distances[i][k] + distances[j][m]
                    s3 = distances[i][m] + distances[j][k]
                    second, largest = sorted((s1, s2, s3))[1:]
                    delta = Fraction(largest - second, 2)
                    if delta > max_delta:
                        max_delta = delta
    return GromovHyperbolicityResult(
        hyperbolicity=CanonicalRational.from_fraction(max_delta)
    )
