"""Domain functions for algebraic topology operations."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.topology.edge_paths._models import (
    EdgePathConcatenateResult,
    EdgePathWordResult,
    OrientedEdge,
)


def _reject(*, location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=f"topology.edge_path.{code}",
        message=message,
    )


def _admit_edge_path_word(
    vertex_count: int,
    edges: tuple[tuple[int, int], ...],
    start_vertex: int,
    path: tuple[OrientedEdge, ...],
) -> None:
    for u, v in edges:
        if not (0 <= u < vertex_count and 0 <= v < vertex_count):
            _reject(
                location=("edges",),
                code="edge_vertex_range",
                message="edge vertices must be in 0..vertex_count-1",
            )
    if not 0 <= start_vertex < vertex_count:
        _reject(
            location=("start_vertex",),
            code="start_vertex_range",
            message="start vertex must be in 0..vertex_count-1",
        )
    current = start_vertex
    for step in path:
        if step.edge_index >= len(edges):
            _reject(
                location=("path",),
                code="edge_index_range",
                message="path edge index is outside the graph",
            )
        left, right = edges[step.edge_index]
        source, target = (left, right) if step.orientation == 1 else (right, left)
        if source != current:
            _reject(
                location=("path",),
                code="path_continuity",
                message="oriented edge path is not continuous",
            )
        current = target


def _admit_edge_path_concatenation(
    vertex_count: int,
    path_a: tuple[int, ...],
    path_b: tuple[int, ...],
) -> None:
    if any(not 0 <= v < vertex_count for v in path_a):
        _reject(
            location=("path_a",),
            code="vertex_range",
            message="path_a vertices must be valid",
        )
    if any(not 0 <= v < vertex_count for v in path_b):
        _reject(
            location=("path_b",),
            code="vertex_range",
            message="path_b vertices must be valid",
        )
    if path_a[-1] != path_b[0]:
        _reject(
            location=("path_b",),
            code="concatenation_endpoint",
            message="concatenated paths must share their endpoint",
        )


def edge_path_word(
    vertex_count: int,
    edges: tuple[tuple[int, int], ...],
    start_vertex: int,
    path: tuple[OrientedEdge, ...],
) -> EdgePathWordResult:
    """Compute the free group word for an edge path.

    Each edge in the graph is assigned a generator label e_i.
    Traversing edge i forward adds e_i, backward adds e_i^{-1}.
    """
    _admit_edge_path_word(vertex_count, edges, start_vertex, path)
    word = [
        f"e{step.edge_index + 1}" + ("" if step.orientation == 1 else "^-1")
        for step in path
    ]
    return EdgePathWordResult(
        word=tuple(word),
        length=len(word),
    )


def concatenate_edge_paths(
    vertex_count: int,
    path_a: tuple[int, ...],
    path_b: tuple[int, ...],
) -> EdgePathConcatenateResult:
    """Concatenate two edge paths.

    If the last vertex of path_a equals the first vertex of path_b,
    the concatenation is path_a + path_b[1:], removing the duplicate.
    """
    _admit_edge_path_concatenation(vertex_count, path_a, path_b)
    first = list(path_a)
    second = list(path_b)
    if first and second and first[-1] == second[0]:
        result = first + second[1:]
    else:
        result = first + second
    return EdgePathConcatenateResult(
        path=tuple(result),
        length=len(result),
    )


__all__ = ["concatenate_edge_paths", "edge_path_word"]
