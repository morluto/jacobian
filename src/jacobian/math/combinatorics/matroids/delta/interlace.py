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
MAX_DISTANCE_INTERLACE_TERMS = 64
MAX_DISTANCE_INTERLACE_COEFFICIENT_BITS = 256


class DistanceInterlaceRequest(StrictModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute the complete subset-distance histogram and "
                "sum_X (x - 1)^d_D(X) for a canonical finite delta-matroid. "
                "Admission permits at most "
                f"{MAX_DISTANCE_INTERLACE_WORK} subset-feasible comparisons "
                f"{MAX_DISTANCE_INTERLACE_TERMS} polynomial terms and "
                f"{MAX_DISTANCE_INTERLACE_COEFFICIENT_BITS}-bit coefficients."
            ),
            "admission_limits": {
                "max_subset_feasible_comparisons": MAX_DISTANCE_INTERLACE_WORK,
                "max_polynomial_terms": MAX_DISTANCE_INTERLACE_TERMS,
                "max_coefficient_bits": MAX_DISTANCE_INTERLACE_COEFFICIENT_BITS,
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
    distance_counts: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_DISTANCE_INTERLACE_TERMS
    )
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

    # There are n+1 histogram entries and polynomial terms. Each polynomial
    # coefficient is bounded in magnitude by 2^n subsets times a coefficient
    # of (x-1)^d, itself at most 2^n. Admit those mathematical allocations
    # before constructing either output or visiting any subsets.
    if n + 1 > MAX_DISTANCE_INTERLACE_TERMS:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code="delta_matroid.distance_interlace_terms",
            message=(
                "distance interlace polynomial exceeds the "
                f"{MAX_DISTANCE_INTERLACE_TERMS}-term output bound"
            ),
        )
    coefficient_bits = 2 * n + 1
    if coefficient_bits > MAX_DISTANCE_INTERLACE_COEFFICIENT_BITS:
        raise OperationResourceAdmissionError(
            location=("delta_matroid",),
            code="delta_matroid.distance_interlace_coefficient_bits",
            message=(
                "distance interlace coefficients exceed the "
                f"{MAX_DISTANCE_INTERLACE_COEFFICIENT_BITS}-bit bound"
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
    "MAX_DISTANCE_INTERLACE_COEFFICIENT_BITS",
    "MAX_DISTANCE_INTERLACE_TERMS",
    "MAX_DISTANCE_INTERLACE_WORK",
    "DistanceInterlaceRequest",
    "DistanceInterlaceResult",
    "distance_interlace_polynomial",
]
