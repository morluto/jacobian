from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tests.component.checkers.exact_domain_checker_support import _request
from tests.support.artifacts import canonical_digest as _digest

from jacobian_checkers.graph_distance_matrix import check_graph_distance_matrix


def _checker_request(
    *,
    vertices: list[str],
    edges: list[list[str]],
    targets: list[str],
    distances: list[list[int | None]],
    connected: bool,
) -> dict[str, Any]:
    return _request(
        "graph.distance_matrix.compute",
        "graph.distance-matrix.all-sources-bfs-v3",
        {
            "graph": {
                "graph_schema_version": "1",
                "vertices": vertices,
                "edges": edges,
            }
        },
        {
            "semantics_version": "unweighted-shortest-path-distance-matrix.v3",
            "row_ordering": "SOURCE_VERTEX_LEXICOGRAPHIC_ASCENDING",
            "target_ordering": "TARGET_VERTEX_LEXICOGRAPHIC_ASCENDING",
            "pair_coverage": "ALL_ORDERED_VERTEX_PAIRS",
            "unreachable_representation": "JSON_NULL",
            "target_vertices": targets,
            "rows": [
                {
                    "source_vertex": source,
                    "distances_by_target": dict(zip(targets, row, strict=True)),
                }
                for source, row in zip(targets, distances, strict=True)
            ],
            "connected": connected,
        },
    )


def _path_request() -> dict[str, Any]:
    return _checker_request(
        vertices=["c", "a", "b"],
        edges=[["a", "b"], ["b", "c"]],
        targets=["a", "b", "c"],
        distances=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        connected=True,
    )


def test_labelled_distance_matrix_checker_accepts_boundary_cases() -> None:
    requests = (
        _checker_request(
            vertices=[],
            edges=[],
            targets=[],
            distances=[],
            connected=False,
        ),
        _checker_request(
            vertices=["only"],
            edges=[],
            targets=["only"],
            distances=[[0]],
            connected=True,
        ),
        _checker_request(
            vertices=["c", "a", "b"],
            edges=[["a", "b"]],
            targets=["a", "b", "c"],
            distances=[[0, 1, None], [1, 0, None], [None, None, 0]],
            connected=False,
        ),
    )

    for request in requests:
        decision = check_graph_distance_matrix(request)
        assert decision["accepted"] is True
        assert decision["conclusion"] == "TRUE"
        assert decision["method"] == "EXHAUSTIVE_FINITE"
        assert decision["coverage"] == "EXHAUSTIVE"


def test_labelled_distance_matrix_checker_rejects_rebound_or_false_values() -> None:
    mutations: tuple[Callable[[dict[str, Any]], object], ...] = (
        lambda result: result.update(target_vertices=["b", "a", "c"]),
        lambda result: result["rows"][0].update(source_vertex="b"),
        lambda result: result["rows"][0]["distances_by_target"].__setitem__(
            "b", True
        ),
        lambda result: result["rows"][0]["distances_by_target"].__setitem__(
            "c", 1
        ),
        lambda result: result.update(connected=False),
        lambda result: result.update(
            semantics_version="unweighted-shortest-path-distance-matrix.v1"
        ),
    )

    for mutate in mutations:
        request = _path_request()
        candidate = request["candidate"]["payload"]
        mutate(candidate)
        request["candidate"]["payload_digest"] = _digest(candidate)

        decision = check_graph_distance_matrix(request)
        assert decision["accepted"] is False
        assert decision["conclusion"] == "UNKNOWN"
