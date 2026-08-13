"""Proof-critical graph cases for the distance-matrix order boundary."""

from __future__ import annotations

from itertools import combinations


def _canonical_edge(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def hoffman_singleton_graph() -> dict[str, object]:
    """Return the standard 50-vertex Hoffman--Singleton construction."""

    vertices = sorted(
        [f"P_{index}_{offset}" for index in range(5) for offset in range(5)]
        + [f"Q_{index}_{offset}" for index in range(5) for offset in range(5)]
    )
    edges: set[tuple[str, str]] = set()
    for index in range(5):
        for offset in range(5):
            edges.add(
                _canonical_edge(
                    f"P_{index}_{offset}",
                    f"P_{index}_{(offset + 1) % 5}",
                )
            )
            edges.add(
                _canonical_edge(
                    f"Q_{index}_{offset}",
                    f"Q_{index}_{(offset + 2) % 5}",
                )
            )
    for left_index in range(5):
        for left_offset in range(5):
            for right_index in range(5):
                edges.add(
                    _canonical_edge(
                        f"P_{left_index}_{left_offset}",
                        f"Q_{right_index}_{(left_index * right_index + left_offset) % 5}",
                    )
                )
    return {
        "graph_schema_version": "1",
        "vertices": vertices,
        "edges": [list(edge) for edge in sorted(edges)],
    }


def _cycle_seven_distance(left: int, right: int) -> int:
    difference = abs(left - right)
    return min(difference, 7 - difference)


def c7_strong_c7_graph() -> dict[str, object]:
    """Return the 49-vertex strong product of two seven-cycles."""

    coordinates = {
        f"{left},{right}": (left, right) for left in range(7) for right in range(7)
    }
    vertices = sorted(coordinates)
    edges = [
        [left, right]
        for left, right in combinations(vertices, 2)
        if max(
            _cycle_seven_distance(
                coordinates[left][0],
                coordinates[right][0],
            ),
            _cycle_seven_distance(
                coordinates[left][1],
                coordinates[right][1],
            ),
        )
        == 1
    ]
    return {
        "graph_schema_version": "1",
        "vertices": vertices,
        "edges": edges,
    }


def c7_strong_c7_distance(source: str, target: str) -> int:
    """Evaluate the exact strong-product distance formula."""

    source_left, source_right = (int(value) for value in source.split(","))
    target_left, target_right = (int(value) for value in target.split(","))
    return max(
        _cycle_seven_distance(source_left, target_left),
        _cycle_seven_distance(source_right, target_right),
    )


__all__ = [
    "c7_strong_c7_distance",
    "c7_strong_c7_graph",
    "hoffman_singleton_graph",
]
