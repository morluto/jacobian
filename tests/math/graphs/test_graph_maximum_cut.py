"""Contract tests for bounded exact maximum cut."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

import networkx as nx
import pytest
from pydantic import ValidationError

from jacobian.math.graphs.optimization import _maximum_cut
from jacobian.math.graphs.optimization._maximum_cut import (
    MAXIMUM_CUT_CANDIDATE_PARTITIONS,
    GraphMaximumCutRequest,
    GraphMaximumCutResult,
    compute_maximum_cut,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def _cycle(order: int) -> SimpleUndirectedGraph:
    vertices = tuple(f"{index:02d}" for index in range(order))
    edges = tuple(
        sorted(
            {
                tuple(sorted((vertices[index], vertices[(index + 1) % order])))
                for index in range(order)
            }
        )
    )
    return _graph(vertices, edges)


def _complete(order: int) -> SimpleUndirectedGraph:
    vertices = tuple(f"{index:02d}" for index in range(order))
    return _graph(vertices, tuple(combinations(vertices, 2)))


def _c5_blow_up(class_sizes: tuple[int, int, int, int, int]) -> SimpleUndirectedGraph:
    classes = tuple(
        tuple(f"{class_index}:{offset}" for offset in range(size))
        for class_index, size in enumerate(class_sizes)
    )
    vertices = tuple(vertex for class_ in classes for vertex in class_)
    edges = tuple(
        sorted(
            {
                tuple(sorted((left, right)))
                for class_index, class_ in enumerate(classes)
                for left in class_
                for right in classes[(class_index + 1) % 5]
            }
        )
    )
    return _graph(vertices, edges)


def _validated_result(graph: SimpleUndirectedGraph) -> GraphMaximumCutResult:
    produced = compute_maximum_cut(GraphMaximumCutRequest(graph=graph))
    return GraphMaximumCutResult.model_validate(produced.model_dump(mode="json"))


def _brute_force_value(graph: SimpleUndirectedGraph) -> int:
    if not graph.vertices:
        return 0
    vertex_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    indexed_edges = tuple(
        (vertex_index[left], vertex_index[right]) for left, right in graph.edges
    )
    best = 0
    for mask in range(1 << (len(graph.vertices) - 1)):
        sides = (
            False,
            *(bool(mask & (1 << index)) for index in range(len(graph.vertices) - 1)),
        )
        best = max(
            best,
            sum(sides[left] != sides[right] for left, right in indexed_edges),
        )
    return best


def _assert_cut_invariant(result: GraphMaximumCutResult) -> None:
    left = set(result.left_vertices)
    expected = tuple(
        edge for edge in result.graph.edges if (edge[0] in left) != (edge[1] in left)
    )
    assert result.crossing_edges == expected
    assert result.cut_value == len(expected)
    assert result.lower_bound == result.cut_value == result.upper_bound


def test_empty_graph_retains_its_source_and_exact_zero_cut() -> None:
    graph = _graph((), ())

    result = _validated_result(graph)

    assert result.graph == graph
    assert result.left_vertices == ()
    assert result.right_vertices == ()
    assert result.crossing_edges == ()
    assert result.cut_value == 0
    assert result.optimality_certificate.source_component_count == 0


def test_bipartite_graph_cuts_every_edge_and_preserves_source_axes() -> None:
    graph = _graph(
        ("z", "a", "m", "b"),
        (("a", "z"), ("a", "m"), ("b", "m")),
    )

    result = _validated_result(graph)

    assert result.cut_value == len(graph.edges)
    assert result.crossing_edges == graph.edges
    assert tuple(
        vertex for vertex in graph.vertices if vertex in result.left_vertices
    ) == (result.left_vertices)
    assert tuple(
        vertex for vertex in graph.vertices if vertex in result.right_vertices
    ) == (result.right_vertices)


@pytest.mark.parametrize(
    ("graph", "expected"),
    [
        (_cycle(5), 4),
        (_complete(6), 9),
        (_c5_blow_up((5, 5, 5, 5, 5)), 100),
        (_c5_blow_up((5, 5, 5, 5, 4)), 95),
    ],
)
def test_known_exact_maximum_cut_values(
    graph: SimpleUndirectedGraph, expected: int
) -> None:
    result = _validated_result(graph)

    assert result.cut_value == expected
    _assert_cut_invariant(result)


def test_false_twin_reduction_admits_the_atlas_scale_blow_up() -> None:
    graph = _c5_blow_up((5, 5, 5, 5, 5))

    result = _validated_result(graph)

    assert len(graph.vertices) == 25
    assert len(graph.edges) == 125
    assert result.optimality_certificate.reduced_vertex_count == 5
    assert result.optimality_certificate.candidate_partitions == 16
    assert result.optimality_certificate.candidate_partitions < (
        MAXIMUM_CUT_CANDIDATE_PARTITIONS
    )


def test_maximum_cut_is_additive_over_connected_components() -> None:
    first = _cycle(5)
    second_vertices = ("k0", "k1", "k2", "k3")
    second_edges = tuple(combinations(second_vertices, 2))
    graph = _graph(first.vertices + second_vertices, first.edges + second_edges)

    result = _validated_result(graph)

    assert result.cut_value == 4 + 4
    assert result.optimality_certificate.source_component_count == 2
    _assert_cut_invariant(result)


def test_every_graph_through_order_six_matches_an_independent_oracle() -> None:
    for atlas_graph in nx.graph_atlas_g():
        if len(atlas_graph) > 6:
            continue
        labels = {vertex: str(vertex) for vertex in atlas_graph}
        vertices = tuple(sorted(labels.values()))
        edges = tuple(
            sorted(
                tuple(sorted((labels[left], labels[right])))
                for left, right in atlas_graph.edges
            )
        )
        graph = _graph(vertices, edges)

        result = compute_maximum_cut(GraphMaximumCutRequest(graph=graph))

        assert result.cut_value == _brute_force_value(graph)
        _assert_cut_invariant(result)


def test_complementary_side_orientation_revalidates() -> None:
    result = _validated_result(_cycle(5))
    payload = result.model_dump(mode="json")
    payload["left_vertices"], payload["right_vertices"] = (
        payload["right_vertices"],
        payload["left_vertices"],
    )

    complemented = GraphMaximumCutResult.model_validate(payload)

    assert complemented.cut_value == result.cut_value
    assert complemented.crossing_edges == result.crossing_edges


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["crossing_edges"].pop(),
            "crossing-edge ledger",
        ),
        (
            lambda payload: payload.__setitem__("cut_value", payload["cut_value"] + 1),
            "cut value",
        ),
        (
            lambda payload: payload.__setitem__(
                "lower_bound", payload["lower_bound"] - 1
            ),
            "exact bounds",
        ),
        (
            lambda payload: payload.__setitem__(
                "upper_bound", payload["upper_bound"] + 1
            ),
            "exact bounds",
        ),
        (
            lambda payload: payload["optimality_certificate"].__setitem__(
                "candidate_partitions",
                payload["optimality_certificate"]["candidate_partitions"] + 1,
            ),
            "certificate",
        ),
    ],
)
def test_result_rejects_forged_ledger_values_bounds_and_certificate(
    mutate: Callable[[dict[str, Any]], object], message: str
) -> None:
    payload = _validated_result(_cycle(5)).model_dump(mode="json")
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        GraphMaximumCutResult.model_validate(payload)


def test_result_rejects_mutated_partition_and_source_graph() -> None:
    payload = _validated_result(_cycle(5)).model_dump(mode="json")
    moved = payload["left_vertices"].pop()
    payload["right_vertices"].append(moved)
    source_order = payload["graph"]["vertices"]
    payload["right_vertices"].sort(key=source_order.index)
    with pytest.raises(ValidationError, match="crossing-edge ledger"):
        GraphMaximumCutResult.model_validate(payload)

    payload = _validated_result(_cycle(5)).model_dump(mode="json")
    removed = payload["crossing_edges"][0]
    payload["graph"]["edges"].remove(removed)
    with pytest.raises(ValidationError, match="source graph"):
        GraphMaximumCutResult.model_validate(payload)


def test_approximate_upper_bound_and_lower_valued_cut_cannot_claim_exactness() -> None:
    honest = _validated_result(_cycle(5)).model_dump(mode="json")
    honest["left_vertices"] = ["00"]
    honest["right_vertices"] = ["01", "02", "03", "04"]
    honest["crossing_edges"] = [["00", "01"], ["00", "04"]]
    honest["cut_value"] = 2
    honest["lower_bound"] = 2
    honest["upper_bound"] = 4

    with pytest.raises(ValidationError, match="exact bounds"):
        GraphMaximumCutResult.model_validate(honest)


def test_candidate_boundary_accepts_c21_and_rejects_c23_before_backend() -> None:
    accepted = _validated_result(_cycle(21))
    assert (
        accepted.optimality_certificate.candidate_partitions
        == MAXIMUM_CUT_CANDIDATE_PARTITIONS
    )

    with pytest.raises(ValidationError, match="candidate partitions"):
        GraphMaximumCutRequest(graph=_cycle(23))


def test_edge_update_boundary_accepts_k19_and_rejects_k20_before_backend() -> None:
    GraphMaximumCutRequest(graph=_complete(19))

    with pytest.raises(ValidationError, match="weighted edge updates"):
        GraphMaximumCutRequest(graph=_complete(20))


def test_large_bipartite_graph_is_not_rejected_by_a_coarse_order_cap() -> None:
    vertices = tuple(f"{index:03d}" for index in range(256))
    graph = _graph(
        vertices,
        tuple((vertices[index], vertices[index + 1]) for index in range(255)),
    )

    result = _validated_result(graph)

    assert result.cut_value == 255
    assert result.optimality_certificate.candidate_partitions == 0


def test_projected_result_bytes_are_rejected_before_search() -> None:
    left = "a" * 1_400_000
    right = "b" * 1_400_000
    graph = _graph((left, right), ((left, right),))

    with pytest.raises(ValidationError, match="projected exact result"):
        GraphMaximumCutRequest(graph=graph)


def test_schema_explains_the_non_schema_representable_work_bounds() -> None:
    graph_schema = GraphMaximumCutRequest.model_json_schema()["properties"]["graph"]

    assert "false-twin quotient" in graph_schema["description"]
    assert str(MAXIMUM_CUT_CANDIDATE_PARTITIONS) in graph_schema["description"]
    assert "exact-only" in graph_schema["description"]


def test_bounded_exhaustive_fallback_preserves_an_exact_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_maximum_cut, "MAXIMUM_CUT_Z3_RLIMIT", 1)

    result = _validated_result(_complete(7))

    assert result.cut_value == 12


def test_operation_is_deterministic_on_a_nonunique_optimum() -> None:
    request = GraphMaximumCutRequest(graph=_cycle(5))

    first = compute_maximum_cut(request)
    second = compute_maximum_cut(request)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
