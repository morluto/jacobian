from __future__ import annotations

from itertools import combinations, product

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.matroids._models import (
    LinearMatroid,
    MatroidWeightedIntersectionOptimizationRequest,
    MatroidWeightedIntersectionOptimizationResult,
    MatroidWeightFunction,
)
from jacobian.math.combinatorics.matroids.intersection import (
    maximum_weight_matroid_intersection,
)
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def _matroid(
    columns: tuple[tuple[int, ...], ...], labels: tuple[str, ...]
) -> LinearMatroid:
    row_count = len(columns[0]) if columns else 0
    rows = tuple(
        tuple(columns[column][row] for column in range(len(columns)))
        for row in range(row_count)
    )
    return LinearMatroid(
        matrix=PrimeFieldMatrix(prime=2, entries=rows, columns=len(labels)),
        ground_labels=labels,
    )


def _independent_by_coefficients(
    matroid: LinearMatroid, subset: tuple[int, ...]
) -> bool:
    """Independent exact oracle: no nonzero coefficient vector kills the columns."""
    for coefficients in product(range(matroid.matrix.prime), repeat=len(subset)):
        if not any(coefficients):
            continue
        if all(
            sum(
                coefficients[index] * row[column] for index, column in enumerate(subset)
            )
            % matroid.matrix.prime
            == 0
            for row in matroid.matrix.entries
        ):
            return False
    return True


def _independent_family(matroid: LinearMatroid) -> tuple[tuple[int, ...], ...]:
    n = matroid.ground_size
    return tuple(
        subset
        for size in range(n + 1)
        for subset in combinations(range(n), size)
        if _independent_by_coefficients(matroid, subset)
    )


def _request(
    first: LinearMatroid, second: LinearMatroid, weights: tuple[int, ...]
) -> MatroidWeightedIntersectionOptimizationRequest:
    return MatroidWeightedIntersectionOptimizationRequest(
        first=first,
        second=second,
        weight_function=MatroidWeightFunction(
            ground_axis=first.ground_axis, values=weights
        ),
    )


def test_weighted_intersection_matches_exhaustive_gf2_instances() -> None:
    """Compare all small represented matroid pairs with a coefficient oracle."""
    labels = ("a", "b", "c")
    representatives: dict[tuple[tuple[int, ...], ...], LinearMatroid] = {}
    for flat in product(range(2), repeat=2 * len(labels)):
        columns = tuple(
            (flat[column], flat[len(labels) + column]) for column in range(len(labels))
        )
        matroid = _matroid(columns, labels)
        representatives.setdefault(_independent_family(matroid), matroid)
    families = tuple(representatives)
    matroids = tuple(representatives.values())
    objectives = tuple(product((-1, 0, 1), repeat=len(labels)))

    for first_index, first in enumerate(matroids):
        for second_index, second in enumerate(matroids):
            common = tuple(
                subset
                for subset in families[first_index]
                if subset in set(families[second_index])
            )
            for weights in objectives:
                expected = min(
                    common,
                    key=lambda subset: (
                        -sum(weights[index] for index in subset),
                        len(subset),
                        subset,
                    ),
                )
                result = maximum_weight_matroid_intersection(
                    _request(first, second, weights)
                )
                assert result.common_independent == expected
                assert result.total_weight == sum(weights[index] for index in expected)


def test_finite_unit_slack_reweights_before_stopping() -> None:
    labels = ("a", "b")
    # In the first source, a and b are parallel. In the second, a is a loop
    # and b is the sole nonloop. The initial source maximizer is a, so the
    # source slack is exactly one and must trigger a dual adjustment to reach b.
    first = _matroid(((1,), (1,)), labels)
    second = _matroid(((0,), (1,)), labels)

    result = maximum_weight_matroid_intersection(_request(first, second, (2, 1)))

    assert result.common_independent == (1,)
    assert result.total_weight == 1
    decoded = MatroidWeightedIntersectionOptimizationResult.model_validate_json(
        result.model_dump_json()
    )
    assert decoded == result


def test_empty_common_set_wins_when_all_common_weights_are_nonpositive() -> None:
    labels = ("a", "b")
    first = _matroid(((1,), (1,)), labels)
    second = _matroid(((1,), (1,)), labels)

    result = maximum_weight_matroid_intersection(_request(first, second, (-4, 0)))

    assert result.common_independent == ()
    assert result.total_weight == 0


def test_exact_near_work_boundary_is_accepted() -> None:
    n = 19
    labels = tuple(f"e{index}" for index in range(n))
    rank_one = _matroid(tuple((1,) for _ in labels), labels)
    weights = (99_999_999_999,) * n

    result = maximum_weight_matroid_intersection(_request(rank_one, rank_one, weights))

    assert result.common_independent == (0,)
    assert result.total_weight == weights[0]


def test_resource_admission_precedes_any_rank_kernel_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jacobian.math.combinatorics.matroids.intersection as intersection

    n = 20
    labels = tuple(f"e{index}" for index in range(n))
    dense = _matroid(
        tuple(tuple(int(i == j) for i in range(n)) for j in range(n)), labels
    )
    request = _request(dense, dense, (99_999_999_999,) * n)

    def unexpected_rank(*args: object, **kwargs: object) -> int:
        raise AssertionError("rank backend was called before weighted admission")

    monkeypatch.setattr(intersection, "pf_rank", unexpected_rank)
    with pytest.raises(OperationResourceAdmissionError) as error:
        maximum_weight_matroid_intersection(request)

    assert error.value.errors()[0]["type"] == (
        "matroid.weighted_intersection.optimize.work_bound"
    )
