"""Supported native colored-graph canonicalization."""

from __future__ import annotations

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from jacobian._execution import BackendFailureReason, OperationBackendError
from jacobian.math.graphs.isomorphism._canonicalization import (
    _canonical_vertex_labels,
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


def _convert_canonicalization_output(  # noqa: C901
    source: ColoredUndirectedGraph,
    canonical_graph: object,
    relabeling: object,
) -> ColoredGraphCanonicalizationResult:
    """Admit the kernel's canonical graph and transporter before promotion."""
    try:
        if type(canonical_graph) is not ColoredUndirectedGraph:
            raise ValueError("canonicalization returned the wrong graph carrier")
        target = canonical_graph
        source_vertices = source.graph.vertices
        target_vertices = target.graph.vertices
        expected_vertices = _canonical_vertex_labels(len(source_vertices))
        if target_vertices != expected_vertices:
            raise ValueError("canonicalization returned the wrong canonical axis")
        if len(target.graph.edges) != len(source.graph.edges):
            raise ValueError("canonicalization changed the edge cardinality")
        if target.vertex_colors and len(target.vertex_colors) != len(target_vertices):
            raise ValueError("canonicalization returned partial vertex colors")
        if target.edge_colors and len(target.edge_colors) != len(target.graph.edges):
            raise ValueError("canonicalization returned partial edge colors")
        if type(relabeling) is not tuple or len(relabeling) != len(source_vertices):
            raise ValueError("canonicalization returned the wrong transporter shape")
        pairs: list[GraphRelabelingPair] = []
        mapping: dict[str, str] = {}
        for item in relabeling:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("canonicalization returned a malformed pair")
            source_vertex, canonical_vertex = item
            if type(source_vertex) is not str or type(canonical_vertex) is not str:
                raise ValueError("canonicalization labels must be exact strings")
            if source_vertex in mapping:
                raise ValueError("canonicalization source labels are not unique")
            mapping[source_vertex] = canonical_vertex
            pairs.append(
                GraphRelabelingPair(
                    source_vertex=source_vertex, canonical_vertex=canonical_vertex
                )
            )
        if tuple(pair.source_vertex for pair in pairs) != source_vertices:
            raise ValueError("canonicalization transporter has the wrong source order")
        if set(mapping) != set(source_vertices) or set(mapping.values()) != set(
            target_vertices
        ):
            raise ValueError("canonicalization transporter is not bijective")
        if bool(source.vertex_colors) != bool(target.vertex_colors):
            raise ValueError("canonicalization changed vertex-color presence")
        if source.vertex_colors:
            source_colors = dict(
                zip(source_vertices, source.vertex_colors, strict=True)
            )
            target_colors = dict(
                zip(target_vertices, target.vertex_colors, strict=True)
            )
            if any(
                target_colors[mapping[vertex]] != source_colors[vertex]
                for vertex in source_vertices
            ):
                raise ValueError("canonicalization changed a vertex color")
        if bool(source.edge_colors) != bool(target.edge_colors):
            raise ValueError("canonicalization changed edge-color presence")
        target_edges = {edge: index for index, edge in enumerate(target.graph.edges)}
        source_edges = {edge: index for index, edge in enumerate(source.graph.edges)}
        mapped_edges: set[tuple[str, str]] = set()
        for edge, index in source_edges.items():
            left, right = (mapping[edge[0]], mapping[edge[1]])
            image = (left, right) if left < right else (right, left)
            if image not in target_edges or image in mapped_edges:
                raise ValueError("canonicalization changed an edge")
            mapped_edges.add(image)
            if source.edge_colors and (
                target.edge_colors[target_edges[image]] != source.edge_colors[index]
            ):
                raise ValueError("canonicalization changed an edge color")
        if mapped_edges != set(target.graph.edges):
            raise ValueError("canonicalization dropped or added an edge")
        return ColoredGraphCanonicalizationResult._from_kernel(
            source_graph=source,
            canonical_graph=target,
            relabeling=tuple(pairs),
        )
    except OperationBackendError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc


def _canonicalize_colored_graph(
    graph: ColoredUndirectedGraph,
) -> ColoredGraphCanonicalizationResult:
    """Construct the exact canonical value from one admitted graph value."""

    require_admitted_colored_graph_canonicalization(graph)
    try:
        canonical_graph, relabeling = canonicalize_colored_graph_data(graph)
    except OperationBackendError:
        raise
    except Exception as exc:
        # The backend is an untrusted mathematical boundary: contradictory
        # output must never escape as a caller-visible exception.
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT) from exc
    return _convert_canonicalization_output(graph, canonical_graph, relabeling)


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
    if tuple(pair.source_vertex for pair in claim.relabeling) != (
        source.graph.vertices
    ):
        return False
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
