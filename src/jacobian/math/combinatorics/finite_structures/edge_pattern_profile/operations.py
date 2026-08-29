"""Hypergraph vertex-colouring edge pattern profile kernel."""

from __future__ import annotations

from jacobian.math.combinatorics.finite_structures.edge_pattern_profile._models import (
    EdgePatternEntry,
    EdgePatternProfileResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

__all__ = ["compute_edge_pattern_profile"]


def compute_edge_pattern_profile(
    hypergraph: FiniteHypergraph,
    vertex_colors: dict[str, str],
) -> EdgePatternProfileResult:
    """Return the complete edge-pattern profile of a vertex-coloured hypergraph.

    For each edge, compute the equality partition of its member colours,
    the number of colour blocks, and classify as monochromatic or rainbow.
    """
    entries: list[EdgePatternEntry] = []
    monochromatic: list[str] = []
    rainbow: list[str] = []

    for edge_id, members in hypergraph.edges:
        colors = [vertex_colors[m] for m in members]
        color_to_block: dict[str, int] = {}
        equality_partition: list[int] = []
        color_labels: list[str] = []
        for c in colors:
            if c not in color_to_block:
                color_to_block[c] = len(color_to_block)
                color_labels.append(c)
            equality_partition.append(color_to_block[c])

        num_blocks = len(color_to_block)
        entries.append(
            EdgePatternEntry(
                edge_id=edge_id,
                members=members,
                equality_partition=tuple(equality_partition),
                num_color_blocks=num_blocks,
                color_labels=tuple(color_labels),
            )
        )
        if num_blocks == 1:
            monochromatic.append(edge_id)
        if num_blocks == len(members):
            rainbow.append(edge_id)

    return EdgePatternProfileResult(
        hypergraph=hypergraph,
        vertex_colors=vertex_colors,
        entries=tuple(entries),
        monochromatic_edge_ids=tuple(monochromatic),
        rainbow_edge_ids=tuple(rainbow),
    )
