"""Pinned distance support profile kernel."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import NoReturn

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.exact._models import PointConfiguration
from jacobian.math.geometry.exact.pinned_distance._models import (
    DistanceClass,
    PinnedDistanceEntry,
    PinnedDistanceSupportProfileResult,
)

__all__ = [
    "compute_pinned_distance_support_profile",
    "verify_pinned_distance_support_profile",
]

MAX_DISTANCE_INTERMEDIATE_DIGITS = 4 * MAX_CANONICAL_RATIONAL_DIGITS


@dataclass(frozen=True, slots=True)
class _DistanceClassPlan:
    squared_distance: CanonicalRational
    target_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _EntryPlan:
    source_label: str
    distance_classes: tuple[_DistanceClassPlan, ...]


def _reject(code: str, message: str) -> NoReturn:
    raise OperationDomainValidationError(
        location=("configuration",),
        code=f"pinned_distance.{code}",
        message=message,
    )


def _admit_configuration(configuration: PointConfiguration) -> tuple[_EntryPlan, ...]:
    if not isinstance(configuration, PointConfiguration):
        _reject("invalid_configuration", "configuration must be a point configuration")

    distances_by_source: list[dict[Fraction, list[str]]] = [
        {} for _point in configuration.points
    ]
    canonical_distances: dict[Fraction, CanonicalRational] = {}
    for index, source in enumerate(configuration.points):
        for target_index in range(index + 1, len(configuration.points)):
            target = configuration.points[target_index]
            squared = _squared_distance(source.coordinates, target.coordinates)
            if squared not in canonical_distances:
                try:
                    canonical_distances[squared] = CanonicalRational.from_fraction(
                        squared
                    )
                except ValueError:
                    _reject(
                        "distance_height_bound",
                        "squared-distance values exceed the canonical rational digit bound",
                    )
            distances_by_source[index].setdefault(squared, []).append(target.label)
            distances_by_source[target_index].setdefault(squared, []).append(
                source.label
            )

    plans: list[_EntryPlan] = []
    for source, distances in zip(
        configuration.points, distances_by_source, strict=True
    ):
        classes: list[_DistanceClassPlan] = []
        for squared, labels in sorted(distances.items()):
            classes.append(
                _DistanceClassPlan(
                    squared_distance=canonical_distances[squared],
                    target_labels=tuple(sorted(labels)),
                )
            )
        plans.append(
            _EntryPlan(source_label=source.label, distance_classes=tuple(classes))
        )

    return tuple(plans)


def compute_pinned_distance_support_profile(
    configuration: PointConfiguration,
) -> PinnedDistanceSupportProfileResult:
    """Return the per-point partition of all other points by squared distance.

    For each source point, group all other points by their exact squared
    Euclidean distance, sorted by increasing distance.
    """
    plans = _admit_configuration(configuration)
    entries = [
        PinnedDistanceEntry(
            source_label=entry.source_label,
            distance_classes=tuple(
                DistanceClass(
                    squared_distance=item.squared_distance,
                    target_labels=item.target_labels,
                )
                for item in entry.distance_classes
            ),
        )
        for entry in plans
    ]

    return PinnedDistanceSupportProfileResult(
        configuration=configuration,
        entries=tuple(entries),
    )


def verify_pinned_distance_support_profile(
    claim: PinnedDistanceSupportProfileResult,
) -> bool:
    """Verify a serialized support profile against its retained configuration."""
    try:
        return compute_pinned_distance_support_profile(claim.configuration) == claim
    except (OperationDomainValidationError, ValueError, RuntimeError):
        return False


def _squared_distance(
    coords_a: tuple[CanonicalRational, ...],
    coords_b: tuple[CanonicalRational, ...],
) -> Fraction:
    """Compute the exact squared Euclidean distance."""
    total = Fraction(0)
    for a, b in zip(coords_a, coords_b, strict=True):
        diff = a.as_fraction() - b.as_fraction()
        term = diff * diff
        _require_bounded_fraction(term)
        new_denominator_factor = term.denominator // gcd(
            total.denominator, term.denominator
        )
        if (
            _integer_digits(total.denominator)
            + _integer_digits(new_denominator_factor)
            - 1
            > MAX_DISTANCE_INTERMEDIATE_DIGITS
        ):
            _reject(
                "distance_intermediate_height_bound",
                "squared-distance summation exceeds the canonical intermediate digit bound",
            )
        total += term
        _require_bounded_fraction(total)
    return total


def _integer_digits(value: int) -> int:
    return len(format_canonical_integer(value).lstrip("-"))


def _require_bounded_fraction(value: Fraction) -> None:
    if (
        _integer_digits(value.numerator) > MAX_DISTANCE_INTERMEDIATE_DIGITS
        or _integer_digits(value.denominator) > MAX_DISTANCE_INTERMEDIATE_DIGITS
    ):
        _reject(
            "distance_intermediate_height_bound",
            "squared-distance arithmetic exceeds the canonical intermediate digit bound",
        )
