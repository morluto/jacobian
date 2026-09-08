"""Finite subsets partitioned by affine cosets of a supplied subspace."""

from jacobian.math.geometry.finite.cosets._models import (
    CosetIntersection,
    CosetIntersectionProfile,
    CosetIntersectionSource,
)
from jacobian.math.geometry.finite.cosets.operations import coset_intersection_profile

__all__ = [
    "CosetIntersection",
    "CosetIntersectionProfile",
    "CosetIntersectionSource",
    "coset_intersection_profile",
]
