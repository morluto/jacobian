"""Exact finite metric space kernels."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

__all__ = ["ball", "gromov_hyperbolicity", "metric_profile"]


def metric_profile(
    distances: list[list[int]],
) -> dict[str, Any]:
    """Compute diameter, radius, eccentricities, centers, and periphery."""
    n = len(distances)
    eccentricities = [max(distances[i]) for i in range(n)]
    diameter = max(eccentricities)
    radius = min(eccentricities)
    centers = tuple(i for i, e in enumerate(eccentricities) if e == radius)
    periphery = tuple(i for i, e in enumerate(eccentricities) if e == diameter)
    return {
        "diameter": diameter,
        "radius": radius,
        "eccentricities": eccentricities,
        "centers": centers,
        "periphery": periphery,
    }


def ball(distances: list[list[int]], center: int, radius: int) -> list[int]:
    """Return the list of points within radius of center."""
    n = len(distances)
    return [i for i in range(n) if distances[center][i] <= radius]


def gromov_hyperbolicity(distances: list[list[int]]) -> Fraction:
    """Compute the four-point Gromov hyperbolicity (max over all quadruples).

    For four points i, j, k, l, define:
    delta(i,j,k,l) = max(0, (largest - second_largest) / 2)
    where largest and second_largest are the two biggest of the three sums
    s1 = d(i,j)+d(k,l), s2 = d(i,k)+d(j,l), s3 = d(i,l)+d(j,k).
    The hyperbolicity is the maximum over all quadruples.
    Since the gap between the two largest sums can be odd, the delta
    can be a half-integer, so we return an exact Fraction.
    """
    n = len(distances)
    max_delta = Fraction(0)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for m in range(k + 1, n):
                    d_ij = distances[i][j]
                    d_kl = distances[k][m]
                    d_ik = distances[i][k]
                    d_jl = distances[j][m]
                    d_il = distances[i][m]
                    d_jk = distances[j][k]
                    s1 = d_ij + d_kl
                    s2 = d_ik + d_jl
                    s3 = d_il + d_jk
                    second, largest = sorted([s1, s2, s3])[1:]
                    delta = Fraction(largest - second, 2)
                    if delta > max_delta:
                        max_delta = delta
    return max_delta
