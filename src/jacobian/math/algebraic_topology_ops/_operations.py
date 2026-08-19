"""Domain functions for algebraic topology operations."""

from __future__ import annotations

from jacobian.math.algebraic_topology_ops._models import (
    EdgePathConcatenateRequest,
    EdgePathConcatenateResult,
    EdgePathWordRequest,
    EdgePathWordResult,
)


def compute_edge_path_word(request: EdgePathWordRequest) -> EdgePathWordResult:
    """Compute the free group word for an edge path.

    Each edge in the graph is assigned a generator label e_i.
    Traversing edge i forward adds e_i, backward adds e_i^{-1}.
    """
    edges = list(request.edges)
    path = list(request.path)
    word: list[str] = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        found = False
        for j, (eu, ev) in enumerate(edges):
            if u == eu and v == ev:
                word.append(f"e{j + 1}")
                found = True
                break
            if u == ev and v == eu:
                word.append(f"e{j + 1}^-1")
                found = True
                break
        if not found:
            raise ValueError(f"path step {u}->{v} is not an edge in the graph")
    return EdgePathWordResult(
        word=tuple(word),
        length=len(word),
    )


def compute_edge_path_concatenate(
    request: EdgePathConcatenateRequest,
) -> EdgePathConcatenateResult:
    """Concatenate two edge paths.

    If the last vertex of path_a equals the first vertex of path_b,
    the concatenation is path_a + path_b[1:], removing the duplicate.
    """
    path_a = list(request.path_a)
    path_b = list(request.path_b)
    if path_a and path_b and path_a[-1] == path_b[0]:
        result = path_a + path_b[1:]
    else:
        result = path_a + path_b
    return EdgePathConcatenateResult(
        path=tuple(result),
        length=len(result),
    )
