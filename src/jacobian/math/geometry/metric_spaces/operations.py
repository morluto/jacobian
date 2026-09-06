"""Exact finite metric space kernels."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError

from ._models import (
    BallResult,
    EccentricityResult,
    FiniteMetricSpace,
    GromovHyperbolicityResult,
    MetricProfileResult,
)

__all__ = ["ball", "gromov_hyperbolicity", "metric_profile", "verify_ball", "verify_gromov_hyperbolicity", "verify_metric_profile"]


def _admit_metric_space(space: FiniteMetricSpace) -> None:
    """Check metric axioms in at most 64 cubed comparisons of 54-bit sums."""

    def reject(reason: str, message: str) -> None:
        raise OperationDomainValidationError(
            location=("metric_space", "distances"),
            code=f"finite_metric_space.{reason}",
            message=message,
        )

    distances = space.distances
    for i in range(space.point_count):
        if distances[i][i] != 0:
            reject("distance_diagonal_nonzero", "diagonal distances must be zero")
        for j in range(space.point_count):
            if distances[i][j] != distances[j][i]:
                reject(
                    "distance_matrix_asymmetric", "distance matrix must be symmetric"
                )
            if i != j and distances[i][j] == 0:
                reject(
                    "distance_nonpositive_between_distinct_points",
                    "distinct points must have positive distance",
                )
            for k in range(space.point_count):
                if distances[i][j] > distances[i][k] + distances[k][j]:
                    reject(
                        "distance_triangle_inequality_violation",
                        "distances must satisfy the triangle inequality",
                    )


def metric_profile(
    metric_space: FiniteMetricSpace,
) -> MetricProfileResult:
    """Compute diameter, radius, eccentricities, centers, and periphery."""
    _admit_metric_space(metric_space)
    distances = metric_space.distances
    n = len(distances)
    eccentricities = [max(distances[i]) for i in range(n)]
    diameter = max(eccentricities)
    radius = min(eccentricities)
    centers = tuple(i for i, e in enumerate(eccentricities) if e == radius)
    periphery = tuple(i for i, e in enumerate(eccentricities) if e == diameter)
    return MetricProfileResult(
        metric_space=metric_space,
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
    _admit_metric_space(metric_space)
    if not 0 <= center < metric_space.point_count:
        raise ValueError("center index must be within the metric space")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    distances = metric_space.distances
    n = len(distances)
    return BallResult(
        metric_space=metric_space,
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
    _admit_metric_space(metric_space)
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
        metric_space=metric_space,
        hyperbolicity=CanonicalRational.from_fraction(max_delta)
    )


def verify_metric_profile(claim: MetricProfileResult) -> bool:
    try:
        return metric_profile(claim.metric_space) == claim
    except (OperationDomainValidationError, TypeError, ValueError):
        return False


def verify_ball(claim: BallResult) -> bool:
    try:
        return ball(claim.metric_space, claim.center, claim.radius) == claim
    except (OperationDomainValidationError, TypeError, ValueError):
        return False


def verify_gromov_hyperbolicity(claim: GromovHyperbolicityResult) -> bool:
    try:
        return gromov_hyperbolicity(claim.metric_space) == claim
    except (OperationDomainValidationError, TypeError, ValueError):
        return False
