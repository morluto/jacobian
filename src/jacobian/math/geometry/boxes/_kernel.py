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
    """Enumerate every nonempty intersection and its inclusion-exclusion sum.

    Intersections are built by a subset dynamic program: removing one selected
    box from a subset leaves a previously computed intersection, so each subset
    needs only one two-box intersection instead of rescanning all of its boxes.
    Emitting masks by cardinality and then lexicographic source position keeps
    the public ledger order identical to the combinations-based definition.
    """

    records: list[IntersectionRecord] = []
    union_volume = Fraction()
    indexed_nonempty = tuple(
        (index, box) for index, box in enumerate(boxes) if not box.is_empty
    )
    active_count = len(indexed_nonempty)
    intersections: list[RationalAxisAlignedBox | None] = [None] * (1 << active_count)
    for mask in range(1, 1 << active_count):
        bit = mask & -mask
        position = bit.bit_length() - 1
        parent = mask ^ bit
        if parent == 0:
            candidate = indexed_nonempty[position][1]
        else:
            parent_intersection = intersections[parent]
            if parent_intersection is None:
                continue
            candidate = intersect_boxes(
                (parent_intersection, indexed_nonempty[position][1])
            )
        if candidate.is_empty:
            continue
        intersections[mask] = candidate

    for subset_size in range(1, active_count + 1):
        coefficient = 1 if subset_size % 2 else -1
        for selected_positions in combinations(range(active_count), subset_size):
            mask = sum(1 << position for position in selected_positions)
            intersection = intersections[mask]
            if intersection is None:
                continue
            box_indices = tuple(
                indexed_nonempty[position][0] for position in selected_positions
            )
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
