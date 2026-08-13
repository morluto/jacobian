"""Independent source-labelled all-pairs distance-matrix replay."""

from __future__ import annotations

from collections import deque
from typing import Any

from jacobian_checkers.bound_artifacts import bound_request

_MAX_ORDER = 32
_MAX_LABEL_LENGTH = 256
_RESULT_KEYS = {
    "semantics_version",
    "row_ordering",
    "target_ordering",
    "pair_coverage",
    "unreachable_representation",
    "target_vertices",
    "rows",
    "connected",
}


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _accept(detail: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "EXHAUSTIVE_FINITE",
        "coverage": "EXHAUSTIVE",
        "detail": detail,
    }


def _finite_simple_graph(
    source: object,
) -> tuple[tuple[str, ...], dict[str, set[str]]]:
    if not isinstance(source, dict) or set(source) != {"graph"}:
        raise ValueError("distance-matrix request is malformed")
    graph = source["graph"]
    if not isinstance(graph, dict) or set(graph) != {
        "graph_schema_version",
        "vertices",
        "edges",
    }:
        raise ValueError("graph input is malformed")
    vertices = graph["vertices"]
    edges = graph["edges"]
    if (
        graph["graph_schema_version"] != "1"
        or not isinstance(vertices, list)
        or len(vertices) > _MAX_ORDER
        or any(
            not isinstance(vertex, str)
            or not 1 <= len(vertex) <= _MAX_LABEL_LENGTH
            for vertex in vertices
        )
        or len(vertices) != len(set(vertices))
        or not isinstance(edges, list)
        or len(edges) > len(vertices) * (len(vertices) - 1) // 2
    ):
        raise ValueError("graph lies outside the checker scope")

    vertex_set = set(vertices)
    normalized_edges: set[tuple[str, str]] = set()
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or not all(isinstance(endpoint, str) for endpoint in edge)
            or edge[0] == edge[1]
            or edge[0] not in vertex_set
            or edge[1] not in vertex_set
        ):
            raise ValueError("graph edge payload is malformed")
        normalized_edges.add(tuple(sorted((edge[0], edge[1]))))
    if len(normalized_edges) != len(edges):
        raise ValueError("graph edge payload contains duplicates")

    adjacency = {vertex: set[str]() for vertex in vertices}
    for left, right in normalized_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return tuple(vertices), adjacency


def _source_distances(
    adjacency: dict[str, set[str]],
    source: str,
) -> dict[str, int]:
    distances = {source: 0}
    frontier = deque([source])
    while frontier:
        vertex = frontier.popleft()
        for neighbor in sorted(adjacency[vertex]):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[vertex] + 1
            frontier.append(neighbor)
    return distances


def _candidate_is_bounded(
    candidate: object,
    targets: tuple[str, ...],
) -> bool:
    if (
        not isinstance(candidate, dict)
        or set(candidate) != _RESULT_KEYS
        or candidate["semantics_version"]
        != "unweighted-shortest-path-distance-matrix.v3"
        or candidate["row_ordering"]
        != "SOURCE_VERTEX_LEXICOGRAPHIC_ASCENDING"
        or candidate["target_ordering"]
        != "TARGET_VERTEX_LEXICOGRAPHIC_ASCENDING"
        or candidate["pair_coverage"] != "ALL_ORDERED_VERTEX_PAIRS"
        or candidate["unreachable_representation"] != "JSON_NULL"
        or candidate["target_vertices"] != list(targets)
        or type(candidate["connected"]) is not bool
    ):
        return False
    rows = candidate["rows"]
    if not isinstance(rows, list) or len(rows) != len(targets):
        return False
    for source, row in zip(targets, rows, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != {"source_vertex", "distances_by_target"}
            or row["source_vertex"] != source
            or not isinstance(row["distances_by_target"], dict)
            or list(row["distances_by_target"]) != list(targets)
        ):
            return False
        for distance in row["distances_by_target"].values():
            if distance is not None and (
                type(distance) is not int or not 0 <= distance <= _MAX_ORDER - 1
            ):
                return False
    return True


def _expected_result(
    vertices: tuple[str, ...],
    adjacency: dict[str, set[str]],
) -> dict[str, Any]:
    targets = tuple(sorted(vertices))
    rows: list[dict[str, Any]] = []
    connected = bool(targets)
    for source in targets:
        distances = _source_distances(adjacency, source)
        row = {target: distances.get(target) for target in targets}
        connected = connected and all(distance is not None for distance in row.values())
        rows.append(
            {
                "source_vertex": source,
                "distances_by_target": row,
            }
        )
    return {
        "semantics_version": "unweighted-shortest-path-distance-matrix.v3",
        "row_ordering": "SOURCE_VERTEX_LEXICOGRAPHIC_ASCENDING",
        "target_ordering": "TARGET_VERTEX_LEXICOGRAPHIC_ASCENDING",
        "pair_coverage": "ALL_ORDERED_VERTEX_PAIRS",
        "unreachable_representation": "JSON_NULL",
        "target_vertices": list(targets),
        "rows": rows,
        "connected": connected,
    }


def check_graph_distance_matrix(request: object) -> dict[str, Any]:
    try:
        source, candidate = bound_request(
            request,
            operation_id="graph.distance_matrix.compute",
            witness_format="graph.distance-matrix.all-sources-bfs-v3",
        )
        vertices, adjacency = _finite_simple_graph(source)
        targets = tuple(sorted(vertices))
        if not _candidate_is_bounded(candidate, targets):
            return _reject("candidate distance-matrix value is malformed")
        if candidate != _expected_result(vertices, adjacency):
            return _reject(
                "candidate does not match independent source-labelled BFS replay"
            )
        return _accept(
            "independent source-labelled all-sources BFS replay accepted "
            "graph.distance_matrix.compute"
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


__all__ = ["check_graph_distance_matrix"]
