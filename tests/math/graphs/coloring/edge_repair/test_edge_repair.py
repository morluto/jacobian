"""Defining behavior for fixed-precolouring edge repair."""

from __future__ import annotations

import json
from itertools import combinations, product

import pytest
from pydantic import ValidationError

from jacobian.math.graphs.coloring import precoloring_edge_repair
from jacobian.math.graphs.coloring._models import (
    PrecoloringEdgeRepairRequest,
    PrecoloringEdgeRepairResult,
)
from jacobian.math.graphs.coloring._tools import TOOLS, compute_precoloring_edge_repair
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph


def _request(
    edges: tuple[tuple[int, int], ...],
    vertex_count: int,
    colors: int,
    fixed_colors: tuple[tuple[int, int], ...] = (),
) -> PrecoloringEdgeRepairRequest:
    return PrecoloringEdgeRepairRequest(
        graph=IndexedSimpleUndirectedGraph(vertex_count=vertex_count, edges=edges),
        colors=colors,
        fixed_colors=fixed_colors,
    )


def _k4(
    colors: int, fixed_colors: tuple[tuple[int, int], ...] = ()
) -> PrecoloringEdgeRepairRequest:
    return _request(
        tuple((left, right) for left, right in combinations(range(4), 2)),
        4,
        colors,
        fixed_colors,
    )


def _verify(result: PrecoloringEdgeRepairResult) -> None:
    assert result.status == "OPTIMAL"
    assert result.coloring.graph == result.graph
    assert result.coloring.colors == result.colors
    for vertex, color in result.fixed_colors:
        assert result.coloring.coloring[vertex] == color
    monochromatic = tuple(
        index
        for index, (left, right) in enumerate(result.graph.edges)
        if result.coloring.coloring[left] == result.coloring.coloring[right]
    )
    assert result.repaired_edge_indices == monochromatic
    assert result.repaired_edge_count == len(monochromatic)


def test_k4_three_colors_needs_one_repair() -> None:
    result = compute_precoloring_edge_repair(_k4(3))

    _verify(result)
    assert result.repaired_edge_count == 1


def test_fixed_precolouring_changes_the_optimum_and_witness() -> None:
    result = compute_precoloring_edge_repair(_k4(2, ((0, 0), (1, 0))))

    _verify(result)
    assert result.repaired_edge_count == 2


def test_forced_monochromatic_edge_is_repaired_under_one_color() -> None:
    request = _request(((0, 1),), 2, 1)
    result = compute_precoloring_edge_repair(request)

    _verify(result)
    assert result.repaired_edge_indices == (0,)


def test_empty_and_disconnected_graphs_are_canonical() -> None:
    empty = compute_precoloring_edge_repair(_request((), 0, 3))
    disconnected = compute_precoloring_edge_repair(_request((), 3, 3))

    _verify(empty)
    _verify(disconnected)
    assert empty.repaired_edge_count == 0
    assert disconnected.repaired_edge_count == 0


def test_total_precolouring_is_replayed_exactly() -> None:
    fixed = ((0, 0), (1, 0), (2, 1), (3, 2))
    result = compute_precoloring_edge_repair(_k4(3, fixed))

    _verify(result)
    assert result.repaired_edge_count == 1


def test_small_graphs_agree_with_exhaustive_colorings() -> None:
    edges = ((0, 1), (0, 2), (1, 2), (1, 3))
    for colors in (1, 2, 3):
        result = compute_precoloring_edge_repair(_request(edges, 4, colors))
        exhaustive = min(
            sum(coloring[left] == coloring[right] for left, right in edges)
            for coloring in product(range(colors), repeat=4)
        )
        _verify(result)
        assert result.repaired_edge_count == exhaustive


def test_result_rejects_nonminimal_or_unbound_witnesses() -> None:
    result = compute_precoloring_edge_repair(_k4(3))
    payload = json.loads(result.model_dump_json())
    payload["repaired_edge_count"] = 2
    with pytest.raises(ValidationError, match="match"):
        PrecoloringEdgeRepairResult.model_validate(payload)


def test_duplicate_fixed_vertices_fail_closed() -> None:
    with pytest.raises(ValidationError, match="one color per vertex"):
        PrecoloringEdgeRepairRequest(
            graph=IndexedSimpleUndirectedGraph(vertex_count=2, edges=()),
            colors=2,
            fixed_colors=((0, 0), (0, 1)),
        )


def test_public_example_and_serialization_round_trip() -> None:
    operation = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "graph.coloring.precoloring_edge_repair.compute"
    )
    request = operation.request_type.model_validate(operation.examples[0].input)
    result = operation.run(request)
    parsed = operation.result_type.model_validate(json.loads(result.model_dump_json()))

    assert parsed == result
    assert result.repaired_edge_count == 1


def test_native_and_catalog_paths_share_the_result() -> None:
    request = _k4(3)
    native = precoloring_edge_repair(
        request.graph,
        request.colors,
        request.fixed_colors,
        request.solver_conflicts,
    )
    catalog = compute_precoloring_edge_repair(request)

    assert native == catalog
