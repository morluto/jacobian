"""Pinned distance support profile kernel."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import NoReturn

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.exact._models import PointConfiguration
from jacobian.math.geometry.exact.pinned_distance._models import (
    DistanceClass,
    PinnedDistanceEntry,
    PinnedDistanceSupportProfileResult,
)

__all__ = ["compute_pinned_distance_support_profile"]

MAX_RESULT_BYTES = CanonicalLimits().max_output_bytes


def _array_size(item_sizes: list[int]) -> int:
    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


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


def _squared_distance_digit_bound(
    left: tuple[CanonicalRational, ...],
    right: tuple[CanonicalRational, ...],
) -> tuple[int, int]:
    """Return conservative numerator and denominator widths for a square sum."""
    max_difference_numerator = 0
    max_difference_denominator = 0
    for first, second in zip(left, right, strict=True):
        max_difference_numerator = max(
            max_difference_numerator,
            max(
                len(first.num.lstrip("-")) + len(second.den),
                len(second.num.lstrip("-")) + len(first.den),
            )
            + 1,
        )
        max_difference_denominator = max(
            max_difference_denominator,
            len(first.den) + len(second.den),
        )
    squared_denominator = 2 * max_difference_denominator
    return (
        2 * max_difference_numerator
        + (len(left) - 1) * squared_denominator
        + len(str(len(left)))
        + 1,
        len(left) * squared_denominator,
    )


def _maximum_result_bytes(
    configuration: PointConfiguration,
    source_bytes: int,
    distance_numerator_digits: int,
    distance_denominator_digits: int,
) -> int:
    points = configuration.points
    max_label_bytes = max(len(encode_strict_json(point.label)) for point in points)
    rational_bytes = strict_json_object_size(
        (
            ("num", distance_numerator_digits),
            ("den", distance_denominator_digits),
        )
    )
    target_labels_bytes = _array_size([max_label_bytes] * max(len(points) - 1, 0))
    class_bytes = strict_json_object_size(
        (
            ("squared_distance", rational_bytes),
            ("target_labels", target_labels_bytes),
        )
    )
    entry_bytes = strict_json_object_size(
        (
            ("source_label", max_label_bytes),
            ("distance_classes", _array_size([class_bytes] * (len(points) - 1))),
        )
    )
    return strict_json_object_size(
        (
            ("configuration", source_bytes),
            ("entries", _array_size([entry_bytes] * len(points))),
        )
    )


def _admit_configuration(configuration: PointConfiguration) -> tuple[_EntryPlan, ...]:
    if not isinstance(configuration, PointConfiguration):
        _reject("invalid_configuration", "configuration must be a point configuration")
    source_bytes = len(encode_strict_json(configuration.model_dump(mode="json")))
    maximum_numerator_digits = 0
    maximum_denominator_digits = 0
    for index, left in enumerate(configuration.points):
        for right in configuration.points[index + 1 :]:
            numerator_digits, denominator_digits = _squared_distance_digit_bound(
                left.coordinates, right.coordinates
            )
            maximum_numerator_digits = max(maximum_numerator_digits, numerator_digits)
            maximum_denominator_digits = max(
                maximum_denominator_digits, denominator_digits
            )
    if (
        maximum_numerator_digits > MAX_CANONICAL_RATIONAL_DIGITS
        or maximum_denominator_digits > MAX_CANONICAL_RATIONAL_DIGITS
    ):
        _reject(
            "distance_height_bound",
            "squared-distance intermediates exceed the canonical rational digit bound",
        )
    result_bytes = _maximum_result_bytes(
        configuration,
        source_bytes,
        maximum_numerator_digits,
        maximum_denominator_digits,
    )
    if result_bytes > MAX_RESULT_BYTES:
        _reject(
            "result_size_bound",
            f"the complete distance profile exceeds the {MAX_RESULT_BYTES}-byte output bound",
        )

    plans: list[_EntryPlan] = []
    for index, source in enumerate(configuration.points):
        distances: dict[Fraction, list[str]] = {}
        for target_index, target in enumerate(configuration.points):
            if index == target_index:
                continue
            squared = _squared_distance(source.coordinates, target.coordinates)
            distances.setdefault(squared, []).append(target.label)
        classes: list[_DistanceClassPlan] = []
        for squared, labels in sorted(distances.items()):
            try:
                canonical = CanonicalRational.from_fraction(squared)
            except ValueError as exc:
                _reject("distance_height_bound", str(exc))
            classes.append(
                _DistanceClassPlan(
                    squared_distance=canonical,
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
