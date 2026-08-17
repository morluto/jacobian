"""Bounded exact chromatic-number operation."""

from __future__ import annotations

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool
from jacobian.math.graphs.optimization._coloring_models import (
    GraphChromaticNumberOutput,
    GraphChromaticNumberRequest,
)
from jacobian.math.graphs.optimization._operations import (
    build_simple_graph,
    solve_chromatic_number,
)


def _search_chromatic_number(
    request: GraphChromaticNumberRequest,
) -> GraphChromaticNumberOutput:
    """Run bounded k-colorability decisions until exactness or timeout."""

    networkx_graph = build_simple_graph(request.graph)
    output = solve_chromatic_number(
        networkx_graph,
        graph=request.graph,
        vertices=request.graph.vertices,
        wall_seconds=request.resource_budget.wall_seconds,
    )

    return output


CHROMATIC_NUMBER_OPERATION = MathTool(
    operation_id="graph.invariant.chromatic_number.compute",
    version="1",
    title="Exact chromatic number",
    description=(
        "Compute the exact chromatic number of a bounded simple undirected "
        "graph by bounded Z3 k-colorability decisions. A timeout returns "
        "an UNKNOWN result with the tested bounds and search trace."
    ),
    request_type=GraphChromaticNumberRequest,
    result_type=GraphChromaticNumberOutput,
    run=_search_chromatic_number,
    tags=(
        "graph",
        "invariant",
        "chromatic_number",
        "exact",
        "bounded",
        "z3",
    ),
    examples=(
        example(
            "triangle_chromatic_number",
            "Compute a triangle's chromatic number (3); vertices must be unique and edges must not self-loop.",
            {
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                }
            },
        ),
    ),
)
