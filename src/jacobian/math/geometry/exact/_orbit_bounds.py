"""Derived admission for exact point-configuration orbit canonicalization."""

from __future__ import annotations

from math import factorial
from typing import NoReturn

from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.exact._models import PointConfiguration

MAX_ORBIT_PERMUTATIONS = 40_320
MAX_ORBIT_DISTANCE_DIGITS = 4_096


def _reject(message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("configuration",),
        code="point_configuration.orbit_work_bound",
        message=message,
    )


def _rational_digits(value: int) -> int:
    return len(format_canonical_integer(abs(value)))


def admit_orbit_profile(configuration: PointConfiguration) -> None:
    """Admit one source before materializing pairwise or permutation work."""
    permutation_count = factorial(len(configuration.points))
    if permutation_count > MAX_ORBIT_PERMUTATIONS:
        _reject("configuration exceeds the admitted permutation-work budget")

    maximum_component_digits = 1
    for point in configuration.points:
        for coordinate in point.coordinates:
            maximum_component_digits = max(
                maximum_component_digits,
                _rational_digits(coordinate.num),
                _rational_digits(coordinate.den),
            )
    pair_count = len(configuration.points) * (len(configuration.points) - 1) // 2
    dimension = len(configuration.points[0].coordinates)
    # A similarity entry is a quotient of two independently bounded squared
    # distances, so its numerator and denominator can each combine both
    # distance envelopes. Charge both returned matrices before execution.
    distance_digits = 4 * dimension * maximum_component_digits + 2
    similarity_digits = 2 * distance_digits
    if pair_count * (distance_digits + similarity_digits) > MAX_ORBIT_DISTANCE_DIGITS:
        _reject("pairwise squared-distance output exceeds the admitted digit budget")
