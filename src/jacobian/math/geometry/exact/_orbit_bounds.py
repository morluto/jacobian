"""Derived admission for exact point-configuration orbit canonicalization."""

from __future__ import annotations

from math import factorial
from typing import NoReturn

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


def _rational_digits(value: str) -> int:
    return len(value.lstrip("-"))


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
    distance_digits = 2 * maximum_component_digits + 1
    if pair_count * distance_digits > MAX_ORBIT_DISTANCE_DIGITS:
        _reject("pairwise squared-distance output exceeds the admitted digit budget")
