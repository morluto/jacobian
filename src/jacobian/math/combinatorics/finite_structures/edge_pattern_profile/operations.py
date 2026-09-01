"""Hypergraph vertex-colouring edge pattern profile kernel."""

from __future__ import annotations

import unicodedata

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.edge_pattern_profile._models import (
    EdgePatternEntry,
    EdgePatternProfileResult,
    VertexColorPair,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

__all__ = [
    "EdgePatternEntry",
    "EdgePatternProfileResult",
    "VertexColorPair",
    "compute_edge_pattern_profile",
]

def _compute_edge_admission(
    edge_id: str,
    members: tuple[str, ...],
    colors: tuple[str, ...],
) -> tuple[tuple[int, ...], int, tuple[str, ...], bool, bool]:
    """Compute one edge's equality partition and classification for admission.

    Returns the equality partition, block count, labels, and classifications.
    """
    color_to_block: dict[str, int] = {}
    equality_partition: list[int] = []
    color_labels: list[str] = []
    for c in colors:
        if c not in color_to_block:
            color_to_block[c] = len(color_to_block)
            color_labels.append(c)
        equality_partition.append(color_to_block[c])

    num_blocks = len(color_to_block)
    is_mono = num_blocks <= 1
    is_rainbow = num_blocks == len(members)

    return (
        tuple(equality_partition),
        num_blocks,
        tuple(color_labels),
        is_mono,
        is_rainbow,
    )


def _admit_edge_pattern_profile(
    hypergraph: FiniteHypergraph, vertex_colors: dict[str, str]
) -> tuple[
    dict[str, str],
    list[
        tuple[str, tuple[str, ...], tuple[int, ...], int, tuple[str, ...], bool, bool]
    ],
    int,
]:
    """Admit the request and return the precomputed admission plan.

    Returns ``(normalized_colors, edge_plans)`` where
    ``edge_plans`` is a list of per-edge tuples containing
    ``(edge_id, members, equality_partition, num_blocks, color_labels,
    is_monochromatic, is_rainbow)`` that the kernel reuses directly
    without recomputing the equality partition.
    """
    if not isinstance(hypergraph, FiniteHypergraph):
        raise OperationDomainValidationError(
            location=("hypergraph",),
            code="edge_pattern.invalid_hypergraph",
            message="hypergraph must be a FiniteHypergraph value",
        )
    if not isinstance(vertex_colors, dict) or set(vertex_colors) != set(
        hypergraph.vertices
    ):
        raise OperationDomainValidationError(
            location=("vertex_colors",),
            code="edge_pattern.color_map_must_cover_all_vertices",
            message="vertex_colors must cover exactly all declared vertices",
        )
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in vertex_colors.items()
    ):
        raise OperationDomainValidationError(
            location=("vertex_colors",),
            code="edge_pattern.invalid_color_map",
            message="vertex_colors must be a string-to-string mapping",
        )
    # Thread 3: Use a cheap aggregate raw UTF-8 bound instead of a fixed
    # per-label ceiling, followed by result-sensitive output admission.
    try:
        for value in vertex_colors.values():
            value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise OperationDomainValidationError(
            location=("vertex_colors",),
            code="edge_pattern.invalid_color_encoding",
            message="vertex color labels must be valid UTF-8",
        ) from exc
    normalized_keys = [unicodedata.normalize("NFC", key) for key in vertex_colors]
    if len(set(normalized_keys)) != len(normalized_keys):
        raise OperationDomainValidationError(
            location=("vertex_colors",),
            code="edge_pattern.color_map_key_collision",
            message="vertex_colors keys collide after Unicode normalization",
        )
    normalized_colors = {
        key: unicodedata.normalize("NFC", value) for key, value in vertex_colors.items()
    }

    # Thread 1: Compute equality partitions during admission and return
    # them as part of the plan so the kernel reuses them without
    # recomputing the same semantic work.
    edge_plans = []
    for edge_id, members in hypergraph.edges:
        colors = tuple(normalized_colors[member] for member in members)
        equality_partition, num_blocks, color_labels, is_mono, is_rainbow = (
            _compute_edge_admission(edge_id, members, colors)
        )
        edge_plans.append(
            (
                edge_id,
                members,
                equality_partition,
                num_blocks,
                color_labels,
                is_mono,
                is_rainbow,
            )
        )
    return normalized_colors, edge_plans


def compute_edge_pattern_profile(
    hypergraph: FiniteHypergraph,
    vertex_colors: dict[str, str],
) -> EdgePatternProfileResult:
    """Return the complete edge-pattern profile of a vertex-coloured hypergraph.

    For each edge, compute the equality partition of its member colours,
    the number of colour blocks, and classify as monochromatic or rainbow.
    """
    normalized_colors, edge_plans = _admit_edge_pattern_profile(
        hypergraph, vertex_colors
    )

    # Thread 1: Reuse the precomputed equality partitions from admission
    # instead of recomputing them in the kernel.
    entries: list[EdgePatternEntry] = []
    monochromatic: list[str] = []
    rainbow: list[str] = []

    for (
        edge_id,
        members,
        equality_partition,
        num_blocks,
        color_labels,
        is_mono,
        is_rainbow,
    ) in edge_plans:
        entries.append(
            EdgePatternEntry(
                edge_id=edge_id,
                members=members,
                equality_partition=equality_partition,
                num_color_blocks=num_blocks,
                color_labels=color_labels,
            )
        )
        if is_mono:
            monochromatic.append(edge_id)
        if is_rainbow:
            rainbow.append(edge_id)

    # Thread 2: Use a list of vertex-color pairs to avoid rational ambiguity
    color_pairs = tuple(
        VertexColorPair(vertex=v, color=c) for v, c in sorted(normalized_colors.items())
    )

    return EdgePatternProfileResult(
        hypergraph=hypergraph,
        vertex_colors=color_pairs,
        entries=tuple(entries),
        monochromatic_edge_ids=tuple(monochromatic),
        rainbow_edge_ids=tuple(rainbow),
    )
