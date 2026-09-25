"""Graphic matroids compose with exact linear-matroid operations."""

from __future__ import annotations

import json
from itertools import combinations

import pytest

from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.matroids import (
    LinearMatroid,
    MatroidWeightFunction,
    graphic_matroid,
    matroid_rank,
    maximum_weight_basis_result,
)
from jacobian.math.combinatorics.matroids._models import GraphicMatroidRequest
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.matrices.finite_fields.linear_algebra import PrimeFieldMatrix


def _graph(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(vertices=vertices, edges=edges)


def _rank_by_components(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> int:
    parent = {vertex: vertex for vertex in vertices}

    def find(vertex: str) -> str:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for left, right in edges:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    return len(vertices) - len({find(vertex) for vertex in vertices})


def _restrict_ground_set(
    matroid: LinearMatroid, subset: tuple[int, ...]
) -> LinearMatroid:
    matrix = PrimeFieldMatrix(
        prime=matroid.matrix.prime,
        entries=tuple(
            tuple(row[index] for index in subset) for row in matroid.matrix.entries
        ),
        columns=len(subset),
    )
    return LinearMatroid(
        matrix=matrix,
        ground_labels=tuple(matroid.ground_axis[index] for index in subset),
    )


def test_triangle_incidence_matroid_has_cycle_relation_and_ground_axis() -> None:
    graph = _graph(("c", "b", "a"), (("b", "c"), ("a", "c"), ("a", "b")))
    matroid = graphic_matroid(GraphicMatroidRequest(graph=graph))

    assert matroid.matrix.prime == 2
    assert matroid.ground_labels == ('["a","b"]', '["a","c"]', '["b","c"]')
    assert matroid_rank(matroid) == 2
    assert all(
        matroid_rank(_restrict_ground_set(matroid, subset)) == len(subset)
        for subset in combinations(range(3), 2)
    )
    assert matroid_rank(matroid) < matroid.ground_size


def test_disconnected_graph_basis_is_a_spanning_forest() -> None:
    vertices = ("a", "b", "c", "d", "isolated")
    edges = (("a", "b"), ("b", "c"), ("c", "d"))
    matroid = graphic_matroid(GraphicMatroidRequest(graph=_graph(vertices, edges)))
    assert matroid_rank(matroid) == _rank_by_components(vertices, edges) == 3


def test_graph_input_order_does_not_change_canonical_representation() -> None:
    first = _graph(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))
    second = _graph(("c", "a", "b"), (("b", "c"), ("a", "c"), ("a", "b")))
    left = graphic_matroid(GraphicMatroidRequest(graph=first))
    right = graphic_matroid(GraphicMatroidRequest(graph=second))
    assert left == right


def test_binary_matroid_composes_with_keyed_maximum_weight_basis() -> None:
    graph = _graph(("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")))
    matroid = graphic_matroid(GraphicMatroidRequest(graph=graph))
    weights = MatroidWeightFunction(
        ground_axis=matroid.ground_axis,
        values=(5, 2, 3),
    )
    result = maximum_weight_basis_result(matroid, weights)
    rank = matroid_rank(matroid)
    bases = tuple(
        subset
        for size in range(rank + 1)
        for subset in combinations(range(len(matroid.ground_axis)), size)
        if size == rank
        and size
        == _rank_by_components(
            graph.vertices,
            tuple(tuple(json.loads(matroid.ground_axis[index])) for index in subset),
        )
    )
    optimum = max(sum(weights.values[index] for index in subset) for subset in bases)
    assert result.rank == 2
    assert result.total_weight == optimum


def test_edge_cap_is_rejected_before_incidence_matrix_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.combinatorics.matroids import graphic

    graph = _graph(
        tuple(f"v{index:02}" for index in range(24)),
        tuple(
            (f"v{left:02}", f"v{right:02}")
            for left, right in combinations(range(24), 2)
        ),
    )

    def forbidden(**_kwargs: object) -> None:
        raise AssertionError("matrix construction ran before edge admission")

    monkeypatch.setattr(graphic.PrimeFieldMatrix, "_from_admitted", forbidden)
    with pytest.raises(OperationResourceAdmissionError, match="at most 256"):
        graphic_matroid(GraphicMatroidRequest(graph=graph))


def test_retained_axis_bound_rejects_before_incidence_matrix_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jacobian.math.combinatorics.matroids import graphic
    from jacobian.math.combinatorics.matroids._models import (
        MAX_GROUND_AXIS_CODEPOINTS,
    )

    huge = "x" * (MAX_GROUND_AXIS_CODEPOINTS + 1)
    graph = _graph(("a", huge), (("a", huge),))

    def forbidden(**_kwargs: object) -> None:
        raise AssertionError("matrix construction ran before axis admission")

    monkeypatch.setattr(graphic.PrimeFieldMatrix, "_from_admitted", forbidden)
    with pytest.raises(
        OperationResourceAdmissionError, match="codepoint allocation bound"
    ):
        graphic_matroid(GraphicMatroidRequest(graph=graph))


def test_retained_axis_at_the_codepoint_boundary_is_admitted() -> None:
    from jacobian.math.combinatorics.matroids._models import (
        MAX_GROUND_AXIS_CODEPOINTS,
    )

    # JSON pair syntax contributes eight codepoints beyond the label text.
    long_vertex = "y" * (MAX_GROUND_AXIS_CODEPOINTS - 8)
    graph = _graph(("a", long_vertex), (("a", long_vertex),))
    matroid = graphic_matroid(GraphicMatroidRequest(graph=graph))
    assert matroid.ground_size == 1
    assert json.loads(matroid.ground_axis[0]) == ["a", long_vertex]
