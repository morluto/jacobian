from __future__ import annotations

from itertools import combinations, product

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.matroids._models import (
    LinearMatroid,
    MatroidWeightedIntersectionOptimizationRequest,
    MatroidWeightFunction,
)
from jacobian.math.combinatorics.matroids.cardinality_weighted_intersection import (
    MatroidCardinalityWeightedIntersectionResult,
    maximum_cardinality_weighted_matroid_intersection,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def _matroid(
    rows: tuple[tuple[int, ...], ...], labels: tuple[str, ...]
) -> LinearMatroid:
    return LinearMatroid(
        matrix=PrimeFieldMatrix(prime=2, entries=rows, columns=len(labels)),
        ground_labels=labels,
    )


def _independent_by_coefficients(
    matroid: LinearMatroid, subset: tuple[int, ...]
) -> bool:
    """Independent oracle from the absence of a nonzero column dependency."""
    for coefficients in product(range(matroid.matrix.prime), repeat=len(subset)):
        if not any(coefficients):
            continue
        if all(
            sum(
                coefficients[position] * row[column]
                for position, column in enumerate(subset)
            )
            % matroid.matrix.prime
            == 0
            for row in matroid.matrix.entries
        ):
            return False
    return True


def _oracle(
    first: LinearMatroid,
    second: LinearMatroid,
    weights: tuple[int, ...],
) -> tuple[int, int]:
    candidates = (
        subset
        for size in range(first.ground_size + 1)
        for subset in combinations(range(first.ground_size), size)
        if _independent_by_coefficients(first, subset)
        and _independent_by_coefficients(second, subset)
    )
    return max(
        (
            (len(subset), sum(weights[index] for index in subset))
            for subset in candidates
        ),
        default=(0, 0),
    )


def test_lexicographic_optimum_matches_exhaustive_subset_oracle() -> None:
    labels = ("a", "b", "c")
    fixtures = (
        (
            _matroid(((1, 0, 1), (0, 1, 1)), labels),
            _matroid(((1, 1, 0), (0, 1, 1)), labels),
            (-10, -3, 5),
        ),
        (
            _matroid(((1, 1, 0),), labels),
            _matroid(((0, 1, 1),), labels),
            (4, -7, 3),
        ),
        (
            _matroid(((1, 0, 0), (0, 1, 0)), labels),
            _matroid(((0, 1, 0), (0, 0, 1)), labels),
            (-2, 0, -2),
        ),
    )
    for first, second, weights in fixtures:
        request = MatroidWeightedIntersectionOptimizationRequest(
            first=first,
            second=second,
            weight_function=MatroidWeightFunction(
                ground_axis=first.ground_axis, values=weights
            ),
        )
        result = maximum_cardinality_weighted_matroid_intersection(request)
        assert (result.cardinality, result.total_weight) == _oracle(
            first, second, weights
        )


def test_malformed_constructed_request_raises_domain_error() -> None:
    request = MatroidWeightedIntersectionOptimizationRequest.model_construct(
        first=None,
        second=None,
        weight_function=None,
    )
    with pytest.raises(OperationDomainValidationError):
        maximum_cardinality_weighted_matroid_intersection(request)


def test_negative_weight_cannot_reduce_cardinality_and_result_round_trips() -> None:
    labels = ("a", "b", "c")
    first = _matroid(((1, 0, 1), (0, 1, 1)), labels)
    second = _matroid(((1, 1, 0), (0, 1, 1)), labels)
    weights = (-10, -3, 5)
    result = maximum_cardinality_weighted_matroid_intersection(
        MatroidWeightedIntersectionOptimizationRequest(
            first=first,
            second=second,
            weight_function=MatroidWeightFunction(
                ground_axis=first.ground_axis, values=weights
            ),
        )
    )
    assert result.cardinality == 2
    assert result.total_weight == 2
    assert (
        tuple(
            value - result.cardinality_bonus
            for value in result.optimized.weight_function.values
        )
        == weights
    )
    restored = MatroidCardinalityWeightedIntersectionResult.model_validate_json(
        result.model_dump_json()
    )
    assert restored == result
