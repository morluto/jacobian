"""Domain functions for algebraic topology operations."""

from __future__ import annotations

from jacobian.math.topology.edge_paths._models import (
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
    word = [
        f"e{step.edge_index + 1}" + ("" if step.orientation == 1 else "^-1")
        for step in request.path
    ]
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
