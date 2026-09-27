from math import comb

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.matroids.delta.extra import BinarySymmetricMatrix
from jacobian.math.combinatorics.matroids.delta.extra_ops import binary
from jacobian.math.combinatorics.matroids.delta.interlace import (
    DistanceInterlaceRequest,
    distance_interlace_polynomial,
)
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid


def _nullity(matrix: tuple[tuple[int, ...], ...], indices: tuple[int, ...]) -> int:
    rows = [[matrix[i][j] for j in indices] for i in indices]
    rank = 0
    for column in range(len(indices)):
        pivot = next((row for row in range(rank, len(rows)) if rows[row][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for row in range(rank + 1, len(rows)):
            if rows[row][column]:
                rows[row] = [
                    left ^ right
                    for left, right in zip(rows[row], rows[rank], strict=True)
                ]
        rank += 1
    return len(indices) - rank


def _polynomial_from_histogram(histogram: tuple[int, ...]) -> tuple[int, ...]:
    ascending = [0] * len(histogram)
    for distance, count in enumerate(histogram):
        for degree in range(distance + 1):
            ascending[degree] += (
                count * comb(distance, degree) * (-1 if (distance - degree) % 2 else 1)
            )
    descending = list(reversed(ascending))
    while len(descending) > 1 and descending[0] == 0:
        descending.pop(0)
    return tuple(descending)


def test_distance_interlace_matches_set_family_and_binary_nullity_oracles() -> None:
    entries = ((1, 1, 0), (1, 0, 1), (0, 1, 0))
    matrix = BinarySymmetricMatrix(ground=("a", "b", "c"), entries=entries)
    delta = binary(matrix).delta_matroid
    result = distance_interlace_polynomial(delta)

    feasible = tuple(frozenset(row) for row in delta.feasible)
    set_family_histogram = [0] * 4
    nullity_histogram = [0] * 4
    for mask in range(1 << len(delta.ground)):
        subset = frozenset(i for i in range(3) if mask >> i & 1)
        distance = min(len(subset ^ row) for row in feasible)
        set_family_histogram[distance] += 1
        indices = tuple(sorted(subset))
        nullity_histogram[_nullity(entries, indices)] += 1

    expected_histogram = tuple(set_family_histogram)
    assert expected_histogram == tuple(nullity_histogram)
    assert result.distance_counts == expected_histogram
    assert result.polynomial.coefficients == _polynomial_from_histogram(
        expected_histogram
    )
    assert result.polynomial.coefficients == (3, 2)


def test_distance_interlace_unit_feasible_family_formula() -> None:
    delta = FiniteDeltaMatroid(ground=("a", "b"), feasible=((),))
    result = distance_interlace_polynomial(delta)
    assert result.distance_counts == (1, 2, 1)
    assert result.polynomial.coefficients == (1, 0, 0)
    assert result.formula == "SUM_SUBSETS_(X_MINUS_1)_TO_DISTANCE"


def test_distance_interlace_rejects_work_before_subset_enumeration() -> None:
    delta = FiniteDeltaMatroid(ground=tuple(f"e{i}" for i in range(18)), feasible=((),))
    with pytest.raises(OperationResourceAdmissionError) as error:
        distance_interlace_polynomial(delta)
    assert error.value.errors()[0]["type"] == "delta_matroid.distance_interlace_work"


def test_distance_interlace_request_schema_uses_exact_source() -> None:
    request = DistanceInterlaceRequest.model_validate(
        {"delta_matroid": {"ground": ["a"], "feasible": [[]]}}
    )
    result = distance_interlace_polynomial(request.delta_matroid)
    assert result.source == request.delta_matroid
