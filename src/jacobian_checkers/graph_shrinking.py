"""Independent property checker for simple-graph counterexample reductions."""

from __future__ import annotations

from collections import deque
from typing import Any


def check_non_bipartite_preservation(request: dict[str, Any]) -> dict[str, Any]:
    """Check independently that the reduced graph remains non-bipartite."""

    try:
        if request.get("request_version") != "1":
            return _reject("unsupported request version")
        claim = request["claim"]["payload"]
        reduced = request["reduced"]["payload"]
        preservation = request["preservation"]["payload"]
        if claim["predicate"]["name"] != "graph.property.non_bipartite":
            return _reject("unsupported graph property")
        if preservation["preservation_format"] != (
            "graph.property.non_bipartite.preservation"
        ):
            return _reject("unexpected preservation format")
        if preservation["format_version"] != "1":
            return _reject("unsupported preservation version")
        if preservation["bindings"] != request["expected_bindings"]:
            return _reject("preservation bindings do not match the request")
        graph = _parse_simple_graph(reduced)
        if graph is None:
            return _reject("reduced graph is not a finite simple undirected graph")
        if _is_bipartite(graph):
            return _reject("reduced graph no longer satisfies non-bipartiteness")
        return {
            "accepted": True,
            "conclusion": "TRUE",
            "arithmetic": "EXACT_INTEGER",
            "method": "EXHAUSTIVE_FINITE",
            "coverage": "EXHAUSTIVE",
            "detail": "reduced graph remains non-bipartite",
        }
    except (KeyError, TypeError, ValueError):
        return _reject("malformed graph-property preservation request")


def _parse_simple_graph(payload: dict[str, Any]) -> dict[str, set[str]] | None:
    vertices = payload.get("vertices")
    edges = payload.get("edges")
    if (
        payload.get("graph_schema_version") != "1"
        or not isinstance(vertices, list)
        or not all(isinstance(vertex, str) and vertex for vertex in vertices)
        or len(vertices) != len(set(vertices))
        or vertices != sorted(vertices)
        or not isinstance(edges, list)
    ):
        return None
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in vertices}
    seen: set[tuple[str, str]] = set()
    for raw_edge in edges:
        if (
            not isinstance(raw_edge, list)
            or len(raw_edge) != 2
            or not all(isinstance(vertex, str) for vertex in raw_edge)
        ):
            return None
        left, right = raw_edge
        edge = (left, right)
        if (
            left >= right
            or left not in adjacency
            or right not in adjacency
            or edge in seen
        ):
            return None
        seen.add(edge)
        adjacency[left].add(right)
        adjacency[right].add(left)
    if edges != [list(edge) for edge in sorted(seen)]:
        return None
    return adjacency


def _is_bipartite(adjacency: dict[str, set[str]]) -> bool:
    colors: dict[str, bool] = {}
    for start in sorted(adjacency):
        if start in colors:
            continue
        colors[start] = False
        queue = deque((start,))
        while queue:
            vertex = queue.popleft()
            for neighbor in sorted(adjacency[vertex]):
                if neighbor not in colors:
                    colors[neighbor] = not colors[vertex]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[vertex]:
                    return False
    return True


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "EXHAUSTIVE_FINITE",
        "coverage": "EXHAUSTIVE",
        "detail": detail,
    }
