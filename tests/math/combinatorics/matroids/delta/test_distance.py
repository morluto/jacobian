"""Exact subset-to-feasibility distance with independent set oracle."""

from itertools import combinations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.matroids.delta import distance
from jacobian.math.combinatorics.matroids.delta._tools import TOOLS
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid


def _subsets(size: int):
    return (
        subset
        for cardinality in range(size + 1)
        for subset in combinations(range(size), cardinality)
    )


def test_distance_matches_independent_hamming_oracle_for_every_ground_subset() -> None:
    delta = FiniteDeltaMatroid(ground=("a", "b", "unused"), feasible=((), (0, 1)))
    feasible_as_sets = tuple(map(set, delta.feasible))

    for subset in _subsets(len(delta.ground)):
        source = set(subset)
        distances = tuple(len(source ^ feasible) for feasible in feasible_as_sets)
        expected_distance = min(distances)
        expected_nearest = min(
            row
            for row, candidate_distance in zip(delta.feasible, distances, strict=True)
            if candidate_distance == expected_distance
        )
        result = distance(delta, subset)
        assert (result.distance, result.nearest_feasible) == (
            expected_distance,
            expected_nearest,
        )
        assert type(result).model_validate(result.model_dump(mode="json")) == result


def test_distance_catalog_example_runs_through_declared_operation() -> None:
    tool = next(
        tool for tool in TOOLS if tool.operation_id == "delta_matroid.distance.compute"
    )
    request = tool.request_type.model_validate(
        {"delta_matroid": {"ground": ["a"], "feasible": [[0]]}, "subset": []}
    )
    assert tool.run(request) == tool.result_type(
        delta_matroid=request.delta_matroid,
        subset=(),
        distance=1,
        nearest_feasible=(0,),
    )


def test_distance_rejects_oversized_source_before_exchange_replay(monkeypatch) -> None:
    import jacobian.math.combinatorics.matroids.delta.operations as operations

    delta = FiniteDeltaMatroid(
        ground=tuple(f"element-{index}" for index in range(500)), feasible=((),)
    )

    def forbidden(_system):
        pytest.fail("exchange kernel ran before source envelope admission")

    monkeypatch.setattr(operations, "_require_delta_matroid_axiom", forbidden)
    with pytest.raises(OperationResourceAdmissionError):
        # Call through the catalog wrapper so native admission has the public
        # resource-failure contract.
        tool = next(
            tool
            for tool in TOOLS
            if tool.operation_id == "delta_matroid.distance.compute"
        )
        tool.run(
            tool.request_type.model_validate(
                {"delta_matroid": delta.model_dump(mode="json"), "subset": []}
            )
        )
