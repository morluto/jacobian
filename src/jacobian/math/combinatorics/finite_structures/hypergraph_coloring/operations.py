"""Non-monochromatic vertex colouring decision for finite hypergraphs."""

from __future__ import annotations

import time
from itertools import product

from pydantic_core import PydanticCustomError

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraph_coloring._models import (
    ColoringWitness,
    NonmonochromaticColoringResult,
    _validate_coloring_envelope,
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
    try:
        admission = _validate_coloring_envelope(hypergraph, palette_size)
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=(), code=error.type, message=str(error)
        ) from error

    vertices = list(hypergraph.vertices)
    edges = list(hypergraph.edges)

    # An empty or singleton edge is monochromatic under every positive palette;
    # return the exact decision without charging or enumerating all colorings.
    if admission.has_forced_failure:
        return NonmonochromaticColoringResult(
            hypergraph=hypergraph,
            palette_size=palette_size,
            outcome="NOT_COLORABLE",
        )

    if not edges:
        witness = ColoringWitness(assignments=tuple((v, 0) for v in vertices))
        return NonmonochromaticColoringResult(
            hypergraph=hypergraph,
            palette_size=palette_size,
            outcome="COLORABLE",
            witness=witness,
        )

    if admission.has_injective_witness:
        witness = ColoringWitness(
            assignments=tuple((vertex, index) for index, vertex in enumerate(vertices))
        )
        return NonmonochromaticColoringResult(
            hypergraph=hypergraph,
            palette_size=palette_size,
            outcome="COLORABLE",
            witness=witness,
        )

    n = len(vertices)
    execution = current_request_execution()
    if execution is not None and execution.deadline is None:
        bind_request_deadline(time.monotonic() + max(60.0, (palette_size**n * len(edges)) / 100_000))
    for index, coloring in enumerate(product(range(palette_size), repeat=n)):
        if index % 1024 == 0:
            execution = current_request_execution()
            if execution is not None and execution.deadline is not None and time.monotonic() >= execution.deadline:
                raise OperationExecutionTimeoutError(
                    "hypergraph coloring search exceeded its request deadline"
                )
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
        if len(colors) < 2:
            return False
    return True
