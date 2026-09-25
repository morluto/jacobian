"""Exact distance interlace polynomial of a finite delta-matroid."""

from __future__ import annotations

from math import comb
from typing import Literal

from pydantic import ConfigDict, Field

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.matroids.delta.extra_ops import _check
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid
from jacobian.math.polynomials._models import IntegerPolynomial

MAX_DISTANCE_INTERLACE_WORK = 250_000
MAX_DISTANCE_INTERLACE_RESULT_BYTES = 1_000_000


class DistanceInterlaceRequest(StrictModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute the complete subset-distance histogram and "
                "sum_X (x - 1)^d_D(X) for a canonical finite delta-matroid. "
                "Admission permits at most "
                f"{MAX_DISTANCE_INTERLACE_WORK} subset-feasible comparisons "
                f"and {MAX_DISTANCE_INTERLACE_RESULT_BYTES} serialized result bytes."
            ),
            "admission_limits": {
                "max_subset_feasible_comparisons": MAX_DISTANCE_INTERLACE_WORK,
                "max_serialized_result_bytes": MAX_DISTANCE_INTERLACE_RESULT_BYTES,
            },
        }
    )

    delta_matroid: FiniteDeltaMatroid = Field(
        description=(
            "Complete canonical feasible family. It is exhaustively validated "
            "as a delta-matroid before distance work is admitted."
        )
    )


class DistanceInterlaceResult(StrictModel):
    source: FiniteDeltaMatroid
    distance_counts: tuple[int, ...] = Field(min_length=1, max_length=2_049)
    polynomial: IntegerPolynomial
    formula: Literal["SUM_SUBSETS_(X_MINUS_1)_TO_DISTANCE"] = (
        "SUM_SUBSETS_(X_MINUS_1)_TO_DISTANCE"
    )


def distance_interlace_polynomial(
    delta_matroid: FiniteDeltaMatroid,
) -> DistanceInterlaceResult:
    """Return ``sum_X (x - 1)^d_D(X)`` and its complete distance histogram."""

    family = _check(delta_matroid)
    n = len(family.ground)
    subset_count = 1 << n
    row_count = len(family.feasible)
    work = subset_count * row_count * max(1, n)
    if work > MAX_DISTANCE_INTERLACE_WORK:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code="delta_matroid.distance_interlace_work",
            message=(
                "distance interlace polynomial exceeds the "
                f"{MAX_DISTANCE_INTERLACE_WORK}-comparison admission bound"
            ),
        )

    # The coefficient of x^k is bounded by 2^n * 2^n: there are 2^n
    # subsets, and each (x-1)^d coefficient has magnitude at most 2^n.
    # Bound the echoed source from its already-admitted labels and memberships,
    # then account for every histogram and polynomial coefficient before
    # enumerating the subset space.
    label_bytes = sum(len(label.encode("utf-8")) for label in family.ground)
    memberships = sum(len(row) for row in family.feasible)
    source_bytes = 16 * label_bytes + 8 * memberships + 3 * row_count + 128
    coefficient_digits = len(str(1 << (2 * n)))
    count_digits = len(str(subset_count))
    result_bound = (
        source_bytes + (n + 1) * (coefficient_digits + count_digits + 16) + 256
    )
    if result_bound > MAX_DISTANCE_INTERLACE_RESULT_BYTES:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code="delta_matroid.distance_interlace_output",
            message=(
                "distance interlace result exceeds the "
                f"{MAX_DISTANCE_INTERLACE_RESULT_BYTES}-byte admission bound"
            ),
        )

    feasible_masks = tuple(sum(1 << index for index in row) for row in family.feasible)
    distance_counts = [0] * (n + 1)
    for subset in range(subset_count):
        distance = min((subset ^ row).bit_count() for row in feasible_masks)
        distance_counts[distance] += 1

    ascending = [0] * (n + 1)
    for distance, count in enumerate(distance_counts):
        if count == 0:
            continue
        for degree in range(distance + 1):
            sign = -1 if (distance - degree) % 2 else 1
            ascending[degree] += sign * count * comb(distance, degree)
    descending = tuple(reversed(ascending))
    while len(descending) > 1 and descending[0] == 0:
        descending = descending[1:]

    return DistanceInterlaceResult(
        source=delta_matroid,
        distance_counts=tuple(distance_counts),
        polynomial=IntegerPolynomial(coefficients=descending),
    )


__all__ = [
    "MAX_DISTANCE_INTERLACE_RESULT_BYTES",
    "MAX_DISTANCE_INTERLACE_WORK",
    "DistanceInterlaceRequest",
    "DistanceInterlaceResult",
    "distance_interlace_polynomial",
]
