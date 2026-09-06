"""Exact graph realization operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.realization._models import (
    DegreeSequenceRequest,
    DegreeSequenceResult,
    GraphicalityCheckRequest,
    GraphicalityCheckResult,
    GraphRealizationRequest,
    GraphRealizationResult,
    RealizationCheckRequest,
    RealizationCheckResult,
)
from jacobian.math.graphs.realization.operations import (
    degree_sequence_profile,
    graph_realization,
    graphicality_check,
    realization_check,
)


def _run_degree_sequence(request: DegreeSequenceRequest) -> DegreeSequenceResult:
    return degree_sequence_profile(request.sequence)


def _run_graph_realization(request: GraphRealizationRequest) -> GraphRealizationResult:
    return graph_realization(request.sequence)


def _run_graphicality_check(
    request: GraphicalityCheckRequest,
) -> GraphicalityCheckResult:
    return graphicality_check(request.sequence)


def _run_realization_check(request: RealizationCheckRequest) -> RealizationCheckResult:
    return realization_check(request.graph, request.sequence)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.realization.is_graphical.compute",
        title="Determine if a degree sequence is graphical",
        description="Determine whether a degree sequence is realized by a simple graph using the Erdos-Gallai theorem.",
        request_type=DegreeSequenceRequest,
        result_type=DegreeSequenceResult,
        run=_run_degree_sequence,
        tags=("graph", "realization", "graphicality", "exact"),
        examples=(
            OperationExample(
                name="graphical_path",
                description="Check the degree sequence of a four-vertex path.",
                input={"sequence": {"degrees": [1, 2, 2, 1]}},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.realization.graphicality_check.compute",
        title="Explain graphicality with a typed Erdos-Gallai certificate",
        description="Determine whether a degree sequence is graphical and return a typed "
        "certificate identifying either the criterion or a failing condition.",
        request_type=GraphicalityCheckRequest,
        result_type=GraphicalityCheckResult,
        run=_run_graphicality_check,
        tags=("graph", "realization", "graphicality", "certificate", "exact"),
        examples=(
            OperationExample(
                name="non_graphical_odd_sum",
                description="Explain why [3, 3, 3] is not graphical.",
                input={"sequence": {"degrees": [3, 3, 3]}},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.realization.construct.compute",
        title="Construct a simple graph realizing a degree sequence",
        description="Construct a simple undirected graph that realizes a graphical degree "
        "sequence using the Havel-Hakimi algorithm. The result retains the source "
        "degree axis and, when graphical, its explicit indexed graph realization.",
        request_type=GraphRealizationRequest,
        result_type=GraphRealizationResult,
        run=_run_graph_realization,
        tags=("graph", "realization", "construction", "exact"),
        examples=(
            OperationExample(
                name="realize_path",
                description="Construct a simple path on 4 vertices from its degree sequence.",
                input={"sequence": {"degrees": [1, 2, 2, 1]}},
            ),
            OperationExample(
                name="realize_cycle",
                description="Construct a 4-cycle from its degree sequence.",
                input={"sequence": {"degrees": [2, 2, 2, 2]}},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.realization.check.compute",
        title="Verify that a graph realizes a degree sequence",
        description="Verify that a given simple undirected graph (vertex_count + edges) "
        "realizes a degree sequence by computing the graph's vertex degrees and "
        "comparing them to the expected sequence.",
        request_type=RealizationCheckRequest,
        result_type=RealizationCheckResult,
        run=_run_realization_check,
        tags=("graph", "realization", "check", "exact"),
        examples=(
            OperationExample(
                name="valid_realization",
                description="A valid realization of the degree sequence [1, 2, 2, 1].",
                input={
                    "sequence": {"degrees": [1, 2, 2, 1]},
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [1, 2], [2, 3]],
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
