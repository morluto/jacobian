"""Bounded exact chromatic-number operation."""

from __future__ import annotations

from jacobian.contracts.graph_coloring import (
    GraphChromaticNumberOutput,
    GraphChromaticNumberRequest,
)
from jacobian.contracts.operations import OperationDiagnostic
from jacobian.contracts.results import ExecutionStatus
from jacobian.domains.graph_optimization.operations import (
    build_simple_graph,
    solve_chromatic_number,
)
from jacobian.operation_declarations import OperationDeclaration, inline_operation
from jacobian.operations import (
    OperationAbortError,
    OperationRefusalError,
)


def _search_chromatic_number(
    request: GraphChromaticNumberRequest,
) -> GraphChromaticNumberOutput:
    """Run bounded k-colorability decisions until exactness or timeout."""

    try:
        networkx_graph = build_simple_graph(request.graph)
    except (KeyError, ValueError, TypeError) as exc:
        raise OperationRefusalError(
            OperationDiagnostic(
                code="CHROMATIC_NUMBER_GRAPH_NOT_APPLICABLE",
                stage="graph_optimization_precondition",
                message=str(exc),
                hint="Supply a simple undirected graph with unique vertices.",
            )
        ) from exc

    output = solve_chromatic_number(
        networkx_graph,
        graph=request.graph,
        vertices=request.graph.vertices,
        wall_seconds=request.resource_budget.wall_seconds,
    )

    if (
        output.vertices != request.graph.vertices
        or output.order != len(request.graph.vertices)
        or (
            output.coloring is not None
            and (
                set(output.coloring) != set(request.graph.vertices)
                or any(
                    output.coloring[left] == output.coloring[right]
                    for left, right in request.graph.edges
                )
            )
        )
    ):
        raise OperationAbortError(
            ExecutionStatus.ERROR,
            OperationDiagnostic(
                code="CHROMATIC_NUMBER_COLORING_INVALID",
                stage="graph_optimization_postcondition",
                message="The solver returned a coloring that does not separate an edge.",
            ),
        )
    if output.status == "EXACT":
        return output
    return output


CHROMATIC_NUMBER_OPERATION = inline_operation(
    OperationDeclaration(
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
        execute=_search_chromatic_number,
        tags=(
            "graph",
            "invariant",
            "chromatic_number",
            "exact",
            "bounded",
            "z3",
        ),
    )
)
