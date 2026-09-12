from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations, product

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.coloring.equitable_k_coloring._models import (
    EquitableColoringResult,
)
from jacobian.math.graphs.coloring.equitable_k_coloring.operations import (
    decide_equitable_k_coloring,
    verify_equitable_coloring,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(
    vertices: Sequence[str], edges: Sequence[Sequence[str]]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple((edge[0], edge[1]) for edge in edges),
    )


def test_k4_equitable() -> None:
    graph = _graph(
        ["a", "b", "c", "d"],
        [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["b", "d"], ["c", "d"]],
    )
    result = decide_equitable_k_coloring(graph, 4)
    assert result.colorable


def test_path_equitable() -> None:
    graph = _graph(["a", "b", "c", "d"], [["a", "b"], ["b", "c"], ["c", "d"]])
    result = decide_equitable_k_coloring(graph, 2)
    assert result.colorable


def test_k3_not_2_colorable() -> None:
    graph = _graph(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
    result = decide_equitable_k_coloring(graph, 2)
    assert not result.colorable


def test_result_preserves_source() -> None:
    graph = _graph(["a", "b"], [["a", "b"]])
    result = decide_equitable_k_coloring(graph, 2)
    assert result.graph == graph
    assert result.k == 2


def test_nonpositive_palette_is_rejected_before_division() -> None:
    with pytest.raises(OperationDomainValidationError, match="positive palette"):
        decide_equitable_k_coloring(_graph(["a"], []), 0)


def test_large_palette_uses_the_direct_singleton_class_construction() -> None:
    graph = _graph([str(index) for index in range(64)], [])
    result = decide_equitable_k_coloring(graph, 64)

    assert result.colorable
    assert result.coloring is not None
    assert result.coloring.coloring == tuple(range(64))


def test_edgeless_graph_uses_direct_balanced_class_construction() -> None:
    graph = _graph([str(index) for index in range(20)], [])
    result = decide_equitable_k_coloring(graph, 2)

    assert result.colorable
    assert result.coloring is not None
    assert result.coloring.coloring.count(0) == 10
    assert result.coloring.coloring.count(1) == 10


def test_bipartite_path_above_generic_search_bound_is_accepted() -> None:
    vertices = [f"v{index:03d}" for index in range(256)]
    graph = _graph(
        vertices, [[vertices[index], vertices[index + 1]] for index in range(255)]
    )

    result = decide_equitable_k_coloring(graph, 2)

    assert result.colorable
    assert result.coloring is not None
    assert result.coloring.coloring.count(0) == 128
    assert result.coloring.coloring.count(1) == 128
    assert verify_equitable_coloring(result)


def test_bipartite_component_orientations_handle_disconnected_graphs_and_isolates() -> (
    None
):
    graph = _graph(
        ["a", "b", "c", "d", "e", "f"],
        [["a", "b"], ["a", "c"], ["d", "e"]],
    )

    result = decide_equitable_k_coloring(graph, 2)

    assert result.colorable
    assert result.coloring is not None
    assert result.coloring.coloring.count(0) == 3
    assert result.coloring.coloring.count(1) == 3
    assert verify_equitable_coloring(result)


def test_bipartite_but_imbalanced_graph_is_not_equitably_two_colorable() -> None:
    graph = _graph(
        ["a", "b", "c", "d", "e", "f"],
        [["a", "b"], ["a", "c"], ["a", "d"], ["a", "e"], ["a", "f"]],
    )

    result = decide_equitable_k_coloring(graph, 2)

    assert not result.colorable
    assert result.coloring is None


def test_odd_cycle_is_not_two_colorable() -> None:
    graph = _graph(
        ["a", "b", "c", "d", "e"],
        [["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"], ["a", "e"]],
    )

    assert not decide_equitable_k_coloring(graph, 2).colorable


def test_large_nonbipartite_graph_is_decided_without_generic_search() -> None:
    vertices = tuple(f"v{index:03d}" for index in range(256))
    graph = _graph(
        vertices,
        [
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[0], vertices[2]],
            *[[vertices[index], vertices[index + 1]] for index in range(2, 255)],
        ],
    )

    result = decide_equitable_k_coloring(graph, 2)

    assert not result.colorable
    assert result.coloring is None


def test_bipartite_decision_matches_independent_bruteforce_on_all_graphs_through_order_four() -> (
    None
):
    for order in range(5):
        vertices = tuple(chr(ord("a") + index) for index in range(order))
        possible_edges = tuple(combinations(vertices, 2))
        for edge_mask in range(1 << len(possible_edges)):
            graph = SimpleUndirectedGraph(
                vertices=vertices,
                edges=tuple(
                    edge
                    for index, edge in enumerate(possible_edges)
                    if edge_mask & (1 << index)
                ),
            )
            expected = any(
                max(colors.count(0), colors.count(1))
                - min(colors.count(0), colors.count(1))
                <= 1
                and all(
                    colors[vertices.index(left)] != colors[vertices.index(right)]
                    for left, right in graph.edges
                )
                for colors in product((0, 1), repeat=order)
            )
            result = decide_equitable_k_coloring(graph, 2)
            assert result.colorable is expected
            if result.colorable:
                assert verify_equitable_coloring(result)


def test_public_tool_serializes_and_verifies_newly_accepted_boundary() -> None:
    from jacobian.math.graphs.coloring.equitable_k_coloring._models import (
        EquitableColoringRequest,
    )
    from jacobian.math.graphs.coloring.equitable_k_coloring._tools import TOOLS

    vertices = tuple(f"v{index:03d}" for index in range(20))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple((vertices[index], vertices[index + 1]) for index in range(19)),
    )

    request = EquitableColoringRequest(graph=graph, k=2)
    decoded = TOOLS[0].run(request)
    assert decoded.colorable
    assert verify_equitable_coloring(decoded)
    assert (
        EquitableColoringResult.model_validate(decoded.model_dump(mode="json"))
        == decoded
    )


def test_one_edge_k1_is_decided_without_recursion() -> None:
    boundary = _graph([f"{index:04d}" for index in range(256)], [["0000", "0001"]])
    result = decide_equitable_k_coloring(boundary, 1)
    assert result.colorable is False
    assert result.coloring is None
    type(result).model_validate(result.model_dump())

    oversized = _graph([f"{index:04d}" for index in range(1100)], [["0000", "0001"]])
    oversized_result = decide_equitable_k_coloring(oversized, 1)
    assert oversized_result.colorable is False
    type(oversized_result).model_validate(oversized_result.model_dump())


def test_search_depth_bound_rejects_before_backtracking() -> None:
    graph = _graph([f"{index:03d}" for index in range(257)], [["000", "001"]])
    with pytest.raises(OperationDomainValidationError, match="search bound"):
        decide_equitable_k_coloring(graph, 2)
