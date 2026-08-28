"""Bounded exact kernel for finite rational box unions."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from jacobian._exact import CanonicalRational
from jacobian.math.analysis.intervals import ClosedRationalInterval
from jacobian.math.geometry.boxes.values import RationalAxisAlignedBox


@dataclass(frozen=True, slots=True)
class IntersectionRecord:
    """One nonempty indexed intersection in inclusion-exclusion order."""

    box_indices: tuple[int, ...]
    intersection: RationalAxisAlignedBox
    volume: Fraction


def intersect_boxes(
    boxes: tuple[RationalAxisAlignedBox, ...],
) -> RationalAxisAlignedBox:
    """Return the exact intersection of a nonempty same-dimensional family."""

    if not boxes:
        raise ValueError("at least one box is required for intersection")
    dimension = boxes[0].dimension
    if any(box.dimension != dimension for box in boxes):
        raise ValueError("all boxes must have the same dimension")
    if any(box.intervals is None for box in boxes):
        return RationalAxisAlignedBox(dimension=dimension, intervals=None)

    interval_rows = tuple(box.intervals for box in boxes if box.intervals is not None)

    intersection: list[ClosedRationalInterval] = []
    for axis in range(dimension):
        axis_intervals = tuple(intervals[axis] for intervals in interval_rows)
        lower = max(
            axis_intervals, key=lambda interval: interval.lower.as_fraction()
        ).lower
        upper = min(
            axis_intervals, key=lambda interval: interval.upper.as_fraction()
        ).upper
        if lower.as_fraction() > upper.as_fraction():
            return RationalAxisAlignedBox(dimension=dimension, intervals=None)
        intersection.append(ClosedRationalInterval(lower=lower, upper=upper))
    return RationalAxisAlignedBox(dimension=dimension, intervals=tuple(intersection))


def box_volume(box: RationalAxisAlignedBox) -> Fraction:
    """Return the exact Lebesgue volume of one rational box."""

    if box.intervals is None:
        return Fraction()
    volume = Fraction(1)
    for interval in box.intervals:
        volume *= interval.upper.as_fraction() - interval.lower.as_fraction()
    return volume


def complete_intersection_ledger(
    boxes: tuple[RationalAxisAlignedBox, ...],
) -> tuple[tuple[IntersectionRecord, ...], Fraction]:
    """Enumerate every nonempty intersection and its inclusion-exclusion sum."""

    records: list[IntersectionRecord] = []
    union_volume = Fraction()
    indexed_nonempty = tuple(
        (index, box) for index, box in enumerate(boxes) if not box.is_empty
    )
    for subset_size in range(1, len(indexed_nonempty) + 1):
        coefficient = 1 if subset_size % 2 else -1
        for selected in combinations(indexed_nonempty, subset_size):
            box_indices = tuple(index for index, _box in selected)
            intersection = intersect_boxes(tuple(box for _index, box in selected))
            if intersection.is_empty:
                continue
            volume = box_volume(intersection)
            records.append(
                IntersectionRecord(
                    box_indices=box_indices,
                    intersection=intersection,
                    volume=volume,
                )
            )
            union_volume += coefficient * volume
    return tuple(records), union_volume


def wire_rational(value: Fraction) -> CanonicalRational:
    return CanonicalRational.from_fraction(value)


__all__ = [
    "IntersectionRecord",
    "box_volume",
    "complete_intersection_ledger",
    "intersect_boxes",
    "wire_rational",
]
