"""Rainbow subgraph embedding profile kernel."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import perm

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


@dataclass(frozen=True, slots=True)
class _RainbowAdmissionPlan:
    candidate_count: int
    rainbow_possible: bool


def _json_array_size(item_sizes: list[int]) -> int:
    return 2 + max(len(item_sizes) - 1, 0) + sum(item_sizes)


def _repeated_array_size(item_size: int, count: int) -> int:
    return 2 + max(count - 1, 0) + item_size * count


def _admit_rainbow_embedding_profile(
    pattern: SimpleUndirectedGraph,
    host: ColoredUndirectedGraph,
) -> _RainbowAdmissionPlan:
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
    try:
        for label in (*pattern.vertices, *host.graph.vertices, *host.edge_colors):
            label.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise OperationDomainValidationError(
            location=("pattern", "host"),
            code="graph.rainbow_embedding.invalid_unicode_label",
            message="graph and color labels must contain valid Unicode scalar values",
        ) from exc

    candidate_count = 0
    if (
        pattern_order <= host_order
        and len(pattern.edges) <= len(host.graph.edges)
        and not _degree_obstruction(pattern, host)
    ):
        candidate_count = perm(host_order, pattern_order)
    # The kernel builds the host edge set and color map once, then each
    # candidate only checks pattern-edge mappings.  Charge host setup
    # additively, not per candidate.
    host_setup = len(host.graph.edges)
    per_candidate = max(1, len(pattern.edges))
    if host_setup + candidate_count * per_candidate > MAX_RAINBOW_EMBEDDING_WORK:
        raise OperationDomainValidationError(
            location=("pattern",),
            code="graph.rainbow_embedding.work_exceeds_bound",
            message="rainbow embedding enumeration exceeds its exact work bound",
        )
    rainbow_possible = len(set(host.edge_colors)) >= len(pattern.edges)

    return _RainbowAdmissionPlan(candidate_count, rainbow_possible)


def _degree_obstruction(
    pattern: SimpleUndirectedGraph, host: ColoredUndirectedGraph
) -> bool:
    """Return whether a pattern degree cannot occur in the host graph."""

    host_degrees = dict.fromkeys(host.graph.vertices, 0)
    for left, right in host.graph.edges:
        host_degrees[left] += 1
        host_degrees[right] += 1
    maximum_host_degree = max(host_degrees.values(), default=0)
    pattern_degrees = dict.fromkeys(pattern.vertices, 0)
    for left, right in pattern.edges:
        pattern_degrees[left] += 1
        pattern_degrees[right] += 1
    return any(degree > maximum_host_degree for degree in pattern_degrees.values())


def compute_rainbow_embedding_profile(
    pattern: SimpleUndirectedGraph,
    host: ColoredUndirectedGraph,
) -> RainbowEmbeddingResult:
    """Return all rainbow embeddings of a pattern into a coloured host.

    For each injective vertex map from pattern to host, check that every
    pattern edge maps to a host edge. The embedding is rainbow if all
    mapped edges have pairwise distinct colours.
    """
    plan = _admit_rainbow_embedding_profile(pattern, host)
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

    if plan.candidate_count == 0:
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
        if plan.rainbow_possible and len(colors) == len(set(colors)):
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
