"""Rainbow subgraph embedding profile kernel."""

from __future__ import annotations

from itertools import permutations
from math import perm

from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    strict_json_object_size,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.rainbow_embedding._models import (
    MAX_HOST_VERTICES,
    MAX_PATTERN_VERTICES,
    EmbeddingWitness,
    RainbowEmbeddingResult,
)
from jacobian.math.graphs.values import (
    ColoredUndirectedGraph,
    SimpleUndirectedGraph,
)

__all__ = ["compute_rainbow_embedding_profile"]

MAX_RAINBOW_EMBEDDING_WORK = 50_000_000


def _json_array_size(item_sizes: list[int]) -> int:
    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


def _admit_rainbow_embedding_profile(
    pattern: SimpleUndirectedGraph,
    host: ColoredUndirectedGraph,
) -> int:
    pattern_order = len(pattern.vertices)
    host_order = len(host.graph.vertices)
    if pattern_order > MAX_PATTERN_VERTICES:
        raise OperationDomainValidationError(
            location=("pattern", "vertices"),
            code="graph.rainbow_embedding.pattern_vertex_count_exceeds_bound",
            message=f"pattern supports at most {MAX_PATTERN_VERTICES} vertices",
        )
    if host_order > MAX_HOST_VERTICES:
        raise OperationDomainValidationError(
            location=("host", "graph", "vertices"),
            code="graph.rainbow_embedding.host_vertex_count_exceeds_bound",
            message=f"host supports at most {MAX_HOST_VERTICES} vertices",
        )
    if host.graph.edges and not host.edge_colors:
        raise OperationDomainValidationError(
            location=("host", "edge_colors"),
            code="graph.rainbow_embedding.edge_colors_must_cover_edges",
            message="edge_colors must assign one color to every host edge",
        )
    if len(host.edge_colors) not in (0, len(host.graph.edges)):
        raise OperationDomainValidationError(
            location=("host", "edge_colors"),
            code="graph.rainbow_embedding.edge_colors_must_cover_edges",
            message="edge_colors must be empty or align with every host edge",
        )

    candidate_count = (
        perm(host_order, pattern_order)
        if pattern_order <= host_order and len(pattern.edges) <= len(host.graph.edges)
        else 0
    )
    edge_work = max(1, len(pattern.edges) + len(host.graph.edges))
    if candidate_count * edge_work > MAX_RAINBOW_EMBEDDING_WORK:
        raise OperationDomainValidationError(
            location=("pattern",),
            code="graph.rainbow_embedding.work_exceeds_bound",
            message="rainbow embedding enumeration exceeds its exact work bound",
        )

    try:
        pattern_bytes = len(encode_strict_json(pattern.model_dump(mode="json")))
        host_bytes = len(encode_strict_json(host.model_dump(mode="json")))
    except CanonicalizationError as exc:
        raise OperationDomainValidationError(
            location=("pattern", "host"),
            code="graph.rainbow_embedding.result_exceeds_output_bound",
            message="rainbow embedding profile exceeds the canonical output bound",
        ) from exc
    map_item_sizes = [
        _json_array_size(
            [
                len(encode_strict_json(pattern_vertex)),
                len(encode_strict_json(host_vertex)),
            ]
        )
        for pattern_vertex in pattern.vertices
        for host_vertex in host.graph.vertices[:1]
    ]
    # Every witness uses one host label per pattern label. The largest host
    # label is the conservative bound for any assignment.
    if host.graph.vertices:
        max_host_label = max(
            len(encode_strict_json(vertex)) for vertex in host.graph.vertices
        )
        map_item_sizes = [
            _json_array_size([len(encode_strict_json(pattern_vertex)), max_host_label])
            for pattern_vertex in pattern.vertices
        ]
    mapping_bytes = _json_array_size(map_item_sizes)
    max_color = max(
        (len(encode_strict_json(color)) for color in host.edge_colors),
        default=0,
    )
    colors_bytes = _json_array_size([max_color] * len(pattern.edges))
    witness_bytes = strict_json_object_size(
        (
            ("pattern_to_host", mapping_bytes),
            ("edge_color_labels", colors_bytes),
        )
    )
    embeddings_bytes = _json_array_size([witness_bytes] * candidate_count)
    result_bytes = strict_json_object_size(
        (
            ("pattern", pattern_bytes),
            ("host", host_bytes),
            ("embeddings", embeddings_bytes),
            ("total_embeddings", max(1, len(str(candidate_count)))),
            ("rainbow_count", max(1, len(str(candidate_count)))),
        )
    )
    if result_bytes > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("pattern",),
            code="graph.rainbow_embedding.result_exceeds_output_bound",
            message="rainbow embedding profile exceeds the canonical output bound",
        )
    return candidate_count


def compute_rainbow_embedding_profile(
    pattern: SimpleUndirectedGraph,
    host: ColoredUndirectedGraph,
) -> RainbowEmbeddingResult:
    """Return all rainbow embeddings of a pattern into a coloured host.

    For each injective vertex map from pattern to host, check that every
    pattern edge maps to a host edge. The embedding is rainbow if all
    mapped edges have pairwise distinct colours.
    """
    _admit_rainbow_embedding_profile(pattern, host)
    pattern_vertices = list(pattern.vertices)
    host_vertices = list(host.graph.vertices)
    host_edges = set(host.graph.edges)
    edge_colors = {}
    for (a, b), c in zip(host.graph.edges, host.edge_colors, strict=True):
        edge_colors[(a, b)] = c

    embeddings: list[EmbeddingWitness] = []

    if len(pattern_vertices) == 0:
        empty = EmbeddingWitness(pattern_to_host=(), edge_color_labels=())
        return RainbowEmbeddingResult(
            pattern=pattern,
            host=host,
            embeddings=(empty,),
            total_embeddings=1,
            rainbow_count=1,
        )

    if len(pattern_vertices) > len(host_vertices):
        return RainbowEmbeddingResult(
            pattern=pattern,
            host=host,
            embeddings=(),
            total_embeddings=0,
            rainbow_count=0,
        )
    if len(pattern.edges) > len(host.graph.edges):
        return RainbowEmbeddingResult(
            pattern=pattern,
            host=host,
            embeddings=(),
            total_embeddings=0,
            rainbow_count=0,
        )

    total = 0
    rainbow = 0

    for host_assignment in permutations(host_vertices, len(pattern_vertices)):
        vertex_map = dict(zip(pattern_vertices, host_assignment, strict=True))
        valid = True
        colors: list[str] = []
        for a, b in pattern.edges:
            ha, hb = vertex_map[a], vertex_map[b]
            edge = (min(ha, hb), max(ha, hb))
            if edge not in host_edges:
                valid = False
                break
            colors.append(edge_colors[edge])
        if not valid:
            continue
        total += 1
        if len(colors) == len(set(colors)):
            rainbow += 1
            embeddings.append(
                EmbeddingWitness(
                    pattern_to_host=tuple((p, vertex_map[p]) for p in pattern_vertices),
                    edge_color_labels=tuple(colors),
                )
            )

    return RainbowEmbeddingResult(
        pattern=pattern,
        host=host,
        embeddings=tuple(embeddings),
        total_embeddings=total,
        rainbow_count=rainbow,
    )
