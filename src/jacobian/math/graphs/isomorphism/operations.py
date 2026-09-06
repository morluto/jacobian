"""Supported native colored-graph canonicalization."""

from __future__ import annotations

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from jacobian.math.graphs.isomorphism._canonicalization import (
    canonicalize_colored_graph_data,
)
from jacobian.math.graphs.isomorphism._canonicalization_bounds import (
    require_admitted_colored_graph_canonicalization,
)
from jacobian.math.graphs.isomorphism._models import (
    ColoredGraphCanonicalizationResult,
    GraphRelabelingPair,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph


def _canonicalize_colored_graph(
    graph: ColoredUndirectedGraph,
) -> ColoredGraphCanonicalizationResult:
    """Construct the exact canonical value from one admitted graph value."""

    require_admitted_colored_graph_canonicalization(graph)
    canonical_graph, relabeling = canonicalize_colored_graph_data(graph)
    return ColoredGraphCanonicalizationResult._from_kernel(
        source_graph=graph,
        canonical_graph=canonical_graph,
        relabeling=tuple(
            GraphRelabelingPair(source_vertex=source, canonical_vertex=target)
            for source, target in relabeling
        ),
    )


def canonicalize_colored_graph(
    graph: ColoredUndirectedGraph,
) -> ColoredGraphCanonicalizationResult:
    """Return the exact color-preserving canonical form and one relabeling.

    Owner-local admission keeps the same typed outcome as the wire path:
    over-bound graphs raise the public ``ValidationError``, not a core-level
    ``PydanticCustomError``, and no wire request is constructed.
    """

    try:
        return _canonicalize_colored_graph(graph)
    except PydanticCustomError as error:
        raise ValidationError.from_exception_data(
            title="canonicalize_colored_graph",
            line_errors=[{"type": error, "input": graph}],
        ) from error


def verify_colored_graph_canonicalization(
    claim: ColoredGraphCanonicalizationResult,
) -> bool:
    """Check a relabeling preserves vertex colors, edges, and edge colors.

    That the target is the canonical minimum needs enumeration and stays
    the producer's outcome; this bounded check covers only the relabeling
    relation against the retained source and canonical graphs.
    """
    source = claim.source_graph
    canonical = claim.canonical_graph
    forward: dict[str, str] = {}
    for pair in claim.relabeling:
        if pair.source_vertex in forward:
            return False
        forward[pair.source_vertex] = pair.canonical_vertex
    if set(forward) != set(source.graph.vertices):
        return False
    if set(forward.values()) != set(canonical.graph.vertices):
        return False
    if bool(source.vertex_colors) != bool(canonical.vertex_colors):
        return False
    if source.vertex_colors:
        source_color = dict(
            zip(source.graph.vertices, source.vertex_colors, strict=True)
        )
        canonical_color = dict(
            zip(canonical.graph.vertices, canonical.vertex_colors, strict=True)
        )
        if any(
            canonical_color[target] != source_color[source_vertex]
            for source_vertex, target in forward.items()
        ):
            return False
    source_edge_index = {edge: index for index, edge in enumerate(source.graph.edges)}
    canonical_edge_index = {
        edge: index for index, edge in enumerate(canonical.graph.edges)
    }
    if bool(source.edge_colors) != bool(canonical.edge_colors):
        return False
    mapped: set[tuple[str, str]] = set()
    for edge in source.graph.edges:
        left, right = edge
        first, second = forward[left], forward[right]
        image = (first, second) if first < second else (second, first)
        if image not in canonical_edge_index or image in mapped:
            return False
        mapped.add(image)
        if source.edge_colors and (
            canonical.edge_colors[canonical_edge_index[image]]
            != source.edge_colors[source_edge_index[edge]]
        ):
            return False
    return mapped == set(canonical.graph.edges)


__all__ = ["canonicalize_colored_graph", "verify_colored_graph_canonicalization"]
