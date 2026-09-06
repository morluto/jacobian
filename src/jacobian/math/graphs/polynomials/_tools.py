"""Exact graph polynomial operation declarations."""

from typing import Any

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.graphs.polynomials._models import (
    GraphPolynomialRequest,
    GraphPolynomialResult,
    MatchingPolynomialRequest,
    TreeIndependencePolynomialAdmissionError,
    TreeIndependencePolynomialRequest,
    TreeIndependencePolynomialResult,
)
from jacobian.math.graphs.polynomials.operations import (
    _graph_polynomial_result,
    independence_polynomial_coefficients,
)


def _run_tutte(request: GraphPolynomialRequest) -> GraphPolynomialResult:
    return _graph_polynomial_result(request.graph, "TUTTE")


def _run_chromatic(request: GraphPolynomialRequest) -> GraphPolynomialResult:
    return _graph_polynomial_result(request.graph, "CHROMATIC")


def _run_flow(request: GraphPolynomialRequest) -> GraphPolynomialResult:
    return _graph_polynomial_result(request.graph, "FLOW")


def _run_independence(
    request: TreeIndependencePolynomialRequest,
) -> TreeIndependencePolynomialResult:
    try:
        coefficients = independence_polynomial_coefficients(request.graph)
    except TreeIndependencePolynomialAdmissionError as exc:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.polynomial.independence.admission",
            message=str(exc),
        ) from exc
    return TreeIndependencePolynomialResult._from_kernel(
        graph=request.graph,
        coefficients=coefficients,
    )


def _run_matching(request: MatchingPolynomialRequest) -> GraphPolynomialResult:
    return _graph_polynomial_result(request.graph, "MATCHING")


_CYCLE_GRAPH_EXAMPLE: dict[str, Any] = {
    "graph": {
        "vertex_count": 4,
        "edges": [
            [0, 1],
            [1, 2],
            [2, 3],
            [0, 3],
        ],
    },
}

_PATH_GRAPH_EXAMPLE: dict[str, Any] = {
    "graph": {
        "vertex_count": 3,
        "edges": [
            [0, 1],
            [1, 2],
        ],
    },
}

_PATH_TREE_EXAMPLE: dict[str, Any] = {
    "graph": {
        "vertices": ["a", "b", "c", "d"],
        "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
    }
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.polynomial.tutte.compute",
        title="Compute the Tutte polynomial",
        description="Compute the exact Tutte polynomial T_G(x, y) of a finite simple "
        "graph using NetworkX, with structural exponent pairs on the ordered "
        "variable axis (x, y).",
        request_type=GraphPolynomialRequest,
        result_type=GraphPolynomialResult,
        run=_run_tutte,
        tags=("graph", "polynomial", "tutte", "exact"),
        examples=(
            OperationExample(
                name="cycle_graph_c4",
                description="Tutte polynomial of the 4-cycle C4.",
                input=_CYCLE_GRAPH_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="graph.polynomial.chromatic.compute",
        title="Compute the chromatic polynomial",
        description="Compute the exact chromatic polynomial chi_G(x) of a finite simple "
        "graph using NetworkX.",
        request_type=GraphPolynomialRequest,
        result_type=GraphPolynomialResult,
        run=_run_chromatic,
        tags=("graph", "polynomial", "chromatic", "exact"),
        examples=(
            OperationExample(
                name="path_graph_p3",
                description="Chromatic polynomial of the path P3.",
                input=_PATH_GRAPH_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="graph.polynomial.flow.compute",
        title="Compute the flow polynomial",
        description="Compute the exact nowhere-zero flow polynomial F_G(x) of a finite "
        "simple graph, derived from the Tutte polynomial.",
        request_type=GraphPolynomialRequest,
        result_type=GraphPolynomialResult,
        run=_run_flow,
        tags=("graph", "polynomial", "flow", "exact"),
        examples=(
            OperationExample(
                name="cycle_graph_c4_flow",
                description="Flow polynomial of the 4-cycle C4.",
                input=_CYCLE_GRAPH_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="graph.polynomial.independence.compute",
        title="Compute a tree independence polynomial",
        description="Compute the exact independence polynomial whose degree-k coefficient "
        "counts independent vertex sets of cardinality k in one nonempty "
        "finite tree. Return its source-bound dense coefficients, independence "
        "number, total independent-set count, and canonical sparse "
        "RationalPolynomial in QQ[x] after a scalar preflight bounds "
        "convolution work and serialized-result size.",
        request_type=TreeIndependencePolynomialRequest,
        result_type=TreeIndependencePolynomialResult,
        run=_run_independence,
        tags=(
            "graph",
            "polynomial",
            "independence",
            "independent-sets",
            "cardinality-count",
            "tree",
            "exact",
        ),
        examples=(
            OperationExample(
                name="path_tree_p4",
                description=(
                    "Compute the exact independence polynomial of P4; the graph "
                    "must be a nonempty tree within this operation's bounded "
                    "convolution-work and coefficient-digit bounds."
                ),
                input=_PATH_TREE_EXAMPLE,
            ),
        ),
    ),
    MathTool(
        operation_id="graph.polynomial.matching.compute",
        title="Compute the matching polynomial",
        description="Compute the exact matching polynomial M_G(x) of a finite simple "
        "graph by the deletion recurrence on graphs with at most 16 vertices.",
        request_type=MatchingPolynomialRequest,
        result_type=GraphPolynomialResult,
        run=_run_matching,
        tags=("graph", "polynomial", "matching", "exact"),
        examples=(
            OperationExample(
                name="path_graph_p3_matching",
                description="Matching polynomial of the path P3.",
                input=_PATH_GRAPH_EXAMPLE,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
