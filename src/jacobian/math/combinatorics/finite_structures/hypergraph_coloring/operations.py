"""Non-monochromatic vertex colouring decision for finite hypergraphs."""

from __future__ import annotations

from itertools import product

from jacobian.math.combinatorics.finite_structures.hypergraph_coloring._models import (
    ColoringWitness,
    NonmonochromaticColoringResult,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)

__all__ = ["decide_nonmonochromatic_coloring"]


def decide_nonmonochromatic_coloring(
    hypergraph: FiniteHypergraph,
    palette_size: int,
) -> NonmonochromaticColoringResult:
    """Decide whether a hypergraph has a q-colouring with no monochromatic edge.

    For every vertex q-colouring, no hyperedge should be monochromatic.
    Returns COLORABLE with one witness colouring, or NOT_COLORABLE.
    """
    vertices = list(hypergraph.vertices)
    edges = list(hypergraph.edges)

    if not edges:
        witness = ColoringWitness(assignments=tuple((v, 0) for v in vertices))
        return NonmonochromaticColoringResult(
            hypergraph=hypergraph,
            palette_size=palette_size,
            outcome="COLORABLE",
            witness=witness,
        )

    n = len(vertices)
    for coloring in product(range(palette_size), repeat=n):
        if _is_valid_coloring(coloring, edges, vertices):
            assignments = tuple((vertices[i], coloring[i]) for i in range(n))
            witness = ColoringWitness(assignments=assignments)
            return NonmonochromaticColoringResult(
                hypergraph=hypergraph,
                palette_size=palette_size,
                outcome="COLORABLE",
                witness=witness,
            )

    return NonmonochromaticColoringResult(
        hypergraph=hypergraph,
        palette_size=palette_size,
        outcome="NOT_COLORABLE",
    )


def _is_valid_coloring(
    coloring: tuple[int, ...],
    edges: list[tuple[str, tuple[str, ...]]],
    vertices: list[str],
) -> bool:
    vertex_to_color = {vertices[i]: coloring[i] for i in range(len(coloring))}
    for _, members in edges:
        colors = {vertex_to_color[m] for m in members}
        if len(colors) == 1:
            return False
    return True
