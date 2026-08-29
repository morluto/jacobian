"""Binary code distance graph kernel."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.codes.nonlinear.distance_graph._models import (
    MAX_CODE_SIZE,
    BinaryCodeDistanceGraphResult,
)
from jacobian.math.combinatorics.codes.nonlinear.values import ExplicitBinaryCode
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph

__all__ = ["compute_distance_graph"]


def _require_distance_graph_admission(
    source: ExplicitBinaryCode,
    target_distance: int,
) -> None:
    if type(target_distance) is not int or target_distance < 0:
        raise OperationDomainValidationError(
            location=("target_distance",),
            code="code.distance_must_be_nonnegative",
            message="target_distance must be a nonnegative integer",
        )
    if target_distance > source.length:
        raise OperationDomainValidationError(
            location=("target_distance",),
            code="code.distance_exceeds_length",
            message="target_distance must not exceed code length",
        )
    if len(source.codewords) > MAX_CODE_SIZE:
        raise OperationDomainValidationError(
            location=("source", "codewords"),
            code="code.too_many_codewords",
            message=f"at most {MAX_CODE_SIZE} codewords are supported",
        )


def compute_distance_graph(
    source: ExplicitBinaryCode,
    target_distance: int,
) -> BinaryCodeDistanceGraphResult:
    """Construct the graph whose edges join codeword pairs at a given Hamming distance.

    Vertices are the canonical codeword indices (0..M-1). An edge (i,j) exists
    iff the Hamming distance between codewords[i] and codewords[j] equals
    target_distance.
    """
    _require_distance_graph_admission(source, target_distance)
    codewords = source.codewords
    n = len(codewords)
    edges: list[tuple[int, int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = sum(
                1 for a, b in zip(codewords[i], codewords[j], strict=True) if a != b
            )
            if dist == target_distance:
                edges.append((i, j))

    graph = IndexedSimpleUndirectedGraph(
        vertex_count=n,
        edges=tuple(edges),
    )
    return BinaryCodeDistanceGraphResult(
        source=source,
        target_distance=target_distance,
        graph=graph,
        edge_count=len(edges),
    )
