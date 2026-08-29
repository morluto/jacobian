"""Rainbow subgraph embedding profile kernel."""

from __future__ import annotations

from itertools import permutations

from jacobian.math.graphs.rainbow_embedding._models import (
    EmbeddingWitness,
    RainbowEmbeddingResult,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph, SimpleUndirectedGraph

__all__ = ["compute_rainbow_embedding_profile"]


def compute_rainbow_embedding_profile(
    pattern: SimpleUndirectedGraph,
    host: ColoredUndirectedGraph,
) -> RainbowEmbeddingResult:
    """Return all rainbow embeddings of a pattern into a coloured host.

    For each injective vertex map from pattern to host, check that every
    pattern edge maps to a host edge. The embedding is rainbow if all
    mapped edges have pairwise distinct colours.
    """
    pattern_vertices = list(pattern.vertices)
    host_vertices = list(host.graph.vertices)
    host_edges = set(host.graph.edges)
    edge_colors = {}
    for (a, b), c in zip(host.graph.edges, host.edge_colors, strict=True):
        edge_colors[(a, b)] = c

    embeddings: list[EmbeddingWitness] = []

    if len(pattern_vertices) == 0:
        return RainbowEmbeddingResult(
            pattern=pattern,
            host=host,
            embeddings=(),
            total_embeddings=0,
            rainbow_count=0,
        )

    if len(pattern_vertices) > len(host_vertices):
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
