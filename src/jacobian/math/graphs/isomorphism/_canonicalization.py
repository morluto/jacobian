"""Bounded exact kernel for colored-graph canonical labeling."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import permutations
from math import factorial

from jacobian.canonical import encode_strict_json
from jacobian.math.graphs.values import ColoredUndirectedGraph, SimpleUndirectedGraph

MAX_CANONICAL_PERMUTATIONS = factorial(9)
MAX_CANONICAL_REPLAY_WORK = 100_000_000
MAX_CANONICALIZATION_RESULT_BYTES = 10 * 1024 * 1024
"""Aggregate canonical-output budget aligned with Jacobian's 10 MiB JSON limit."""
_RESULT_ENVELOPE_RESERVE_BYTES = 1_024


def _canonical_vertex_labels(vertex_count: int) -> tuple[str, ...]:
    """Return a fixed-width canonical axis whose labels sort in index order."""

    width = max(2, len(str(max(0, vertex_count - 1))))
    return tuple(f"v{index:0{width}d}" for index in range(vertex_count))


def _vertex_classes(graph: ColoredUndirectedGraph) -> tuple[tuple[int, ...], ...]:
    if not graph.vertex_colors:
        return (tuple(range(len(graph.graph.vertices))),)
    classes: dict[str, list[int]] = {}
    for index, color in enumerate(graph.vertex_colors):
        classes.setdefault(color, []).append(index)
    return tuple(tuple(classes[color]) for color in sorted(classes))


def canonical_permutation_count(graph: ColoredUndirectedGraph) -> int:
    """Return the exact number of color-preserving labelings inspected per pass."""

    count = 1
    for vertex_class in _vertex_classes(graph):
        count *= factorial(len(vertex_class))
    return count


def canonical_replay_work(graph: ColoredUndirectedGraph) -> int:
    """Bound execution plus result replay in integer key-work units.

    Each candidate assigns every vertex and transforms every edge.  The
    ``m * m`` term conservatively covers transformed-edge ordering without
    depending on one Python sorting implementation's comparison constant.
    Edge colors are converted to bounded integer ranks before this loop, so
    each comparison has constant-size components.
    """

    vertex_count = len(graph.graph.vertices)
    edge_count = len(graph.graph.edges)
    edge_key_work = edge_count * edge_count
    per_candidate = max(1, vertex_count + edge_key_work)
    return 2 * canonical_permutation_count(graph) * per_candidate


def _target_vectors(
    vertex_classes: tuple[tuple[int, ...], ...],
    vertex_count: int,
) -> Iterator[tuple[int, ...]]:
    targets = [-1] * vertex_count

    def visit(class_index: int, offset: int) -> Iterator[tuple[int, ...]]:
        if class_index == len(vertex_classes):
            yield tuple(targets)
            return
        vertex_class = vertex_classes[class_index]
        for source_order in permutations(vertex_class):
            for position, source_index in enumerate(source_order, start=offset):
                targets[source_index] = position
            yield from visit(class_index + 1, offset + len(vertex_class))

    yield from visit(0, 0)


def _relabel_graph(
    graph: ColoredUndirectedGraph,
    target_by_source: tuple[str, ...],
) -> ColoredUndirectedGraph:
    source_index = {vertex: index for index, vertex in enumerate(graph.graph.vertices)}
    target_vertices = tuple(sorted(target_by_source))

    transformed_vertex_colors: tuple[str, ...] = ()
    if graph.vertex_colors:
        color_by_target = {
            target_by_source[index]: graph.vertex_colors[index]
            for index in range(len(graph.graph.vertices))
        }
        transformed_vertex_colors = tuple(
            color_by_target[vertex] for vertex in target_vertices
        )

    transformed_edges: list[tuple[tuple[str, str], str | None]] = []
    for edge_index, (left, right) in enumerate(graph.graph.edges):
        mapped_left = target_by_source[source_index[left]]
        mapped_right = target_by_source[source_index[right]]
        mapped_edge = (
            (mapped_left, mapped_right)
            if mapped_left < mapped_right
            else (mapped_right, mapped_left)
        )
        color = graph.edge_colors[edge_index] if graph.edge_colors else None
        transformed_edges.append((mapped_edge, color))
    transformed_edges.sort(key=lambda item: item[0])

    return ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=target_vertices,
            edges=tuple(edge for edge, _color in transformed_edges),
        ),
        vertex_colors=transformed_vertex_colors,
        edge_colors=(
            tuple(color for _edge, color in transformed_edges if color is not None)
            if graph.edge_colors
            else ()
        ),
    )


def apply_colored_graph_relabeling(
    graph: ColoredUndirectedGraph,
    relabeling: Mapping[str, str],
) -> ColoredUndirectedGraph:
    """Apply one total vertex relabeling to a canonical colored-graph value."""

    if set(relabeling) != set(graph.graph.vertices):
        raise ValueError("relabeling must map every source vertex exactly once")
    target_by_source = tuple(relabeling[vertex] for vertex in graph.graph.vertices)
    if len(set(target_by_source)) != len(target_by_source):
        raise ValueError("relabeling targets must be unique")
    return _relabel_graph(graph, target_by_source)


def canonicalization_result_wire_bytes(graph: ColoredUndirectedGraph) -> int:
    """Return an exact shape-based upper bound for the public result bytes."""

    canonical_vertices = _canonical_vertex_labels(len(graph.graph.vertices))
    placeholder = _relabel_graph(graph, canonical_vertices)
    payload = {
        "source_graph": graph.model_dump(mode="json"),
        "canonical_graph": placeholder.model_dump(mode="json"),
        "relabeling": [
            {
                "source_vertex": source,
                "canonical_vertex": target,
            }
            for source, target in zip(
                graph.graph.vertices, canonical_vertices, strict=True
            )
        ],
    }
    return len(encode_strict_json(payload)) + _RESULT_ENVELOPE_RESERVE_BYTES


def canonicalize_colored_graph_data(
    graph: ColoredUndirectedGraph,
) -> tuple[ColoredUndirectedGraph, tuple[tuple[str, str], ...]]:
    """Return the lexicographically least colored edge form and transporter.

    Vertex-color classes occupy canonical positions in increasing color-name
    order.  Within those classes, the kernel minimizes the sorted sequence of
    ``(left position, right position, edge-color rank)`` triples.  An
    automorphism tie is resolved by the target-position tuple aligned to the
    source graph's vertex axis.
    """

    source_index = {vertex: index for index, vertex in enumerate(graph.graph.vertices)}
    indexed_edges = tuple(
        (source_index[left], source_index[right]) for left, right in graph.graph.edges
    )
    if graph.edge_colors:
        edge_color_rank = {
            color: rank for rank, color in enumerate(sorted(set(graph.edge_colors)))
        }
        ranked_edge_colors = tuple(
            edge_color_rank[color] for color in graph.edge_colors
        )
        colors_by_rank = tuple(sorted(edge_color_rank))
    else:
        ranked_edge_colors = (-1,) * len(graph.graph.edges)
        colors_by_rank = ()

    vertex_classes = _vertex_classes(graph)
    best_edge_key: tuple[tuple[int, int, int], ...] | None = None
    best_targets: tuple[int, ...] | None = None
    for targets in _target_vectors(vertex_classes, len(graph.graph.vertices)):
        edge_key = tuple(
            sorted(
                (
                    min(targets[left], targets[right]),
                    max(targets[left], targets[right]),
                    ranked_edge_colors[edge_index],
                )
                for edge_index, (left, right) in enumerate(indexed_edges)
            )
        )
        if best_edge_key is None or (edge_key, targets) < (
            best_edge_key,
            best_targets,
        ):
            best_edge_key = edge_key
            best_targets = targets

    if best_edge_key is None or best_targets is None:
        raise AssertionError("finite color classes must yield one canonical labeling")

    canonical_vertices = _canonical_vertex_labels(len(graph.graph.vertices))
    canonical_vertex_colors = (
        tuple(
            graph.vertex_colors[vertex_index]
            for vertex_class in vertex_classes
            for vertex_index in vertex_class
        )
        if graph.vertex_colors
        else ()
    )
    canonical_graph = ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(
            vertices=canonical_vertices,
            edges=tuple(
                (canonical_vertices[left], canonical_vertices[right])
                for left, right, _color in best_edge_key
            ),
        ),
        vertex_colors=canonical_vertex_colors,
        edge_colors=(
            tuple(colors_by_rank[color] for _left, _right, color in best_edge_key)
            if graph.edge_colors
            else ()
        ),
    )
    relabeling = tuple(
        (source, canonical_vertices[best_targets[index]])
        for index, source in enumerate(graph.graph.vertices)
    )
    return canonical_graph, relabeling


__all__ = [
    "MAX_CANONICALIZATION_RESULT_BYTES",
    "MAX_CANONICAL_PERMUTATIONS",
    "MAX_CANONICAL_REPLAY_WORK",
    "apply_colored_graph_relabeling",
    "canonical_permutation_count",
    "canonical_replay_work",
    "canonicalization_result_wire_bytes",
    "canonicalize_colored_graph_data",
]
