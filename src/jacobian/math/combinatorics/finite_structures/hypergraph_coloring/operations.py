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

__all__ = [
    "decide_nonmonochromatic_coloring",
    "verify_coloring_witness",
    "verify_non_colorable",
    "verify_nonmonochromatic_coloring",
]


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

    # Establish the operation-owned deadline before any presolve return so
    # native and dispatched calls cover result construction on every path.
    execution = current_request_execution()
    work_budget = admission.work_budget
    if execution is not None:
        if execution.deadline is None:
            deadline = execution.started_at + max(60.0, work_budget / 100_000)
            bind_request_deadline(deadline)
        else:
            deadline = execution.deadline
    else:
        deadline = time.monotonic() + max(60.0, work_budget / 100_000)

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
    for index, coloring in enumerate(product(range(palette_size), repeat=n)):
        if index % 1024 == 0 and deadline is not None and time.monotonic() >= deadline:
            raise OperationExecutionTimeoutError(
                "hypergraph coloring search exceeded its request deadline"
            )
        if _is_valid_coloring(coloring, edges, vertices, deadline):
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
    deadline: float | None = None,
) -> bool:
    vertex_to_color = {vertices[i]: coloring[i] for i in range(len(coloring))}
    for edge_index, (_, members) in enumerate(edges):
        if (
            edge_index % 256 == 0
            and deadline is not None
            and time.monotonic() >= deadline
        ):
            raise OperationExecutionTimeoutError(
                "hypergraph coloring edge checks exceeded its request deadline"
            )
        colors = {vertex_to_color[m] for m in members}
        if len(colors) < 2:
            return False
    return True


def verify_coloring_witness(claim: NonmonochromaticColoringResult) -> bool:
    """Check a serialized COLORABLE witness against its retained hypergraph."""

    if claim.outcome != "COLORABLE" or claim.witness is None:
        return False
    vertices = tuple(claim.hypergraph.vertices)
    assignments = claim.witness.assignments
    if len(assignments) != len(vertices):
        return False
    if {vertex for vertex, _color in assignments} != set(vertices):
        return False
    if len({vertex for vertex, _color in assignments}) != len(assignments):
        return False
    colors = dict(assignments)
    if any(
        type(color) is not int or not 0 <= color < claim.palette_size
        for color in colors.values()
    ):
        return False
    return all(
        len({colors[vertex] for vertex in members}) >= 2
        for _edge_id, members in claim.hypergraph.edges
    )


def verify_non_colorable(claim: NonmonochromaticColoringResult) -> bool:
    """Re-establish an admitted bounded NOT_COLORABLE decision."""

    if claim.outcome != "NOT_COLORABLE" or claim.witness is not None:
        return False
    try:
        return (
            decide_nonmonochromatic_coloring(
                claim.hypergraph, claim.palette_size
            ).outcome
            == "NOT_COLORABLE"
        )
    except (OperationDomainValidationError, TypeError, ValueError):
        return False


def verify_nonmonochromatic_coloring(
    claim: NonmonochromaticColoringResult,
) -> bool:
    """Verify whichever explicit coloring relation a result claims."""

    if claim.outcome == "COLORABLE":
        return verify_coloring_witness(claim)
    return verify_non_colorable(claim)
