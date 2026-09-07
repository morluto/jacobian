"""Defining behavior for prescribed-degree bipartite factors."""

from __future__ import annotations

import json
from itertools import combinations

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.flows import bipartite_degree_constrained_factor
from jacobian.math.graphs.flows._models import (
    BipartiteFactorRequest,
    BipartiteFactorResult,
)
from jacobian.math.graphs.flows._tools import TOOLS, compute_bipartite_factor
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph


def _request(
    edges: tuple[tuple[int, int], ...],
    left_size: int,
    right_size: int,
    requirements: tuple[int, ...],
) -> BipartiteFactorRequest:
    return BipartiteFactorRequest(
        graph=IndexedSimpleUndirectedGraph(
            vertex_count=left_size + right_size, edges=edges
        ),
        left=tuple(range(left_size)),
        right=tuple(range(left_size, left_size + right_size)),
        required_degrees=requirements,
    )


def _k33(requirements: tuple[int, ...]) -> BipartiteFactorRequest:
    return _request(
        tuple((left, right) for left in range(3) for right in range(3, 6)),
        3,
        3,
        requirements,
    )


def _selected_edges(result: object) -> tuple[tuple[int, int], ...]:
    assert isinstance(result, BipartiteFactorResult)
    return tuple(result.graph.edges[index] for index in result.selected_edge_indices)


def _selected_degrees(edges: tuple[tuple[int, int], ...]) -> list[int]:
    degrees = [0] * 5
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    return degrees


def test_catalog_publishes_one_bipartite_factor_operation() -> None:
    operation_ids = {tool.operation_id for tool in TOOLS}
    assert "graph.bipartite.degree_constrained_factor.compute" in operation_ids


def test_k33_has_a_source_bound_two_regular_factor() -> None:
    result = compute_bipartite_factor(_k33((2, 2, 2, 2, 2, 2)))

    assert result.status == "FOUND"
    selected = _selected_edges(result)
    assert len(selected) == 6
    degrees = [0] * 6
    for left, right in selected:
        degrees[left] += 1
        degrees[right] += 1
    assert degrees == [2] * 6
    assert all(edge in result.graph.edges for edge in selected)
    assert (
        bipartite_degree_constrained_factor(
            result.graph, result.left, result.right, (2, 2, 2, 2, 2, 2)
        )
        == result
    )


def test_unequal_demand_totals_fail_closed_before_flow() -> None:
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_bipartite_factor(_k33((2, 2, 2, 1, 1, 1)))

    assert exc_info.value.errors()[0]["type"] == (
        "graph.bipartite_factor.requirement_totals_disagree"
    )


def test_requirement_above_source_degree_fails_closed() -> None:
    request = _request(
        ((0, 2), (0, 3), (1, 2), (1, 3)),
        2,
        2,
        (3, 0, 0, 0),
    )
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_bipartite_factor(request)

    assert exc_info.value.errors()[0]["type"] == (
        "graph.bipartite_factor.requirement_exceeds_source_degree"
    )


def test_hall_obstruction_is_replayable_and_exact() -> None:
    request = _request(
        ((0, 3), (1, 3), (2, 4)),
        3,
        3,
        (1, 1, 0, 1, 1, 0),
    )
    result = compute_bipartite_factor(request)

    assert result.status == "INFEASIBLE"
    assert result.selected_edge_indices == ()
    assert result.obstruction is not None
    assert result.obstruction.side == "LEFT"
    assert result.obstruction.vertices == (0, 1)
    assert result.obstruction.neighbors == (3,)
    assert result.obstruction.required == 2
    assert result.obstruction.capacity == 1


def test_zero_requirements_and_disconnected_vertices_are_feasible() -> None:
    request = _request(
        ((0, 2),),
        2,
        2,
        (0, 0, 0, 0),
    )
    result = compute_bipartite_factor(request)

    assert result.status == "FOUND"
    assert result.selected_edge_indices == ()
    assert result.requirements.left == (0, 0)
    assert result.requirements.right == (0, 0)


def test_perfect_matching_composes_with_source_edge_indices() -> None:
    request = _request(
        ((0, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 3)),
        3,
        3,
        (1, 1, 1, 1, 1, 1),
    )
    result = compute_bipartite_factor(request)

    assert result.status == "FOUND"
    assert len(result.selected_edge_indices) == 3
    assert len(_selected_edges(result)) == 3


def test_small_cases_agree_with_exhaustive_edge_subsets() -> None:
    edges = ((0, 3), (0, 4), (1, 3), (1, 4), (2, 3))
    left_size, right_size = 3, 2
    for left_subset in (
        left for count in range(4) for left in combinations(range(3), count)
    ):
        for right_subset in (
            right for count in range(3) for right in combinations(range(2), count)
        ):
            degrees = [
                int(vertex in left_subset or vertex in right_subset)
                for vertex in range(5)
            ]
            if degrees[:3] != [int(vertex in left_subset) for vertex in range(3)]:
                continue
            if degrees[3:] != [int(vertex in right_subset) for vertex in range(3, 5)]:
                continue
            if sum(degrees[:3]) != sum(degrees[3:]):
                continue
            request = _request(edges, left_size, right_size, tuple(degrees))
            result = compute_bipartite_factor(request)
            exhaustive = any(
                _selected_degrees(selected) == degrees
                for count in range(len(edges) + 1)
                for selected in combinations(edges, count)
            )
            assert (result.status == "FOUND") == exhaustive


def test_crossing_partition_is_validated_by_typed_request() -> None:
    with pytest.raises(ValidationError, match="cross"):
        BipartiteFactorRequest(
            graph=IndexedSimpleUndirectedGraph(vertex_count=4, edges=((0, 1), (2, 3))),
            left=(0, 1),
            right=(2, 3),
            required_degrees=(0, 0, 0, 0),
        )


def test_requirement_axes_are_source_bound() -> None:
    with pytest.raises(ValidationError, match="at least 2"):
        BipartiteFactorRequest(
            graph=IndexedSimpleUndirectedGraph(vertex_count=2, edges=()),
            left=(0,),
            right=(1,),
            required_degrees=(0,),
        )


def test_result_rejects_cross_outcome_witnesses() -> None:
    request = _k33((0, 0, 0, 0, 0, 0))
    payload = request.model_dump()
    payload.update(
        status="FOUND",
        selected_edge_indices=[0, 0],
        requirements={"left": [0, 0, 0], "right": [0, 0, 0]},
    )
    payload.pop("required_degrees", None)
    with pytest.raises(ValidationError, match="distinct"):
        BipartiteFactorResult.model_validate(payload)


def test_public_example_and_serialization_round_trip() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == ("graph.bipartite.degree_constrained_factor.compute")
    )
    example = operation.examples[0]
    request = operation.request_type.model_validate(example.input)
    result = operation.run(request)
    parsed = operation.result_type.model_validate(json.loads(result.model_dump_json()))

    assert parsed == result
    assert parsed.status == "FOUND"
    assert parsed.requirements.left == (2, 2, 2)
    assert parsed.requirements.right == (2, 2, 2)


def test_admission_rejects_oversized_degree_before_flow() -> None:
    with pytest.raises(ValidationError, match=r"0\.\.1000000000"):
        _request(((0, 1),), 1, 1, (1_000_000_001, 0))
