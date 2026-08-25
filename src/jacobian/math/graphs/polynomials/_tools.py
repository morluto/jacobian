"""Exact graph polynomial operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.polynomials._models import (
    GraphPolynomialRequest,
    GraphPolynomialResult,
    MatchingPolynomialRequest,
    SparseMultivariatePolynomial,
    TreeIndependencePolynomialRequest,
    TreeIndependencePolynomialResult,
)
from jacobian.math.graphs.polynomials._operations import (
    compute_chromatic_polynomial,
    compute_flow_polynomial,
    compute_independence_polynomial,
    compute_matching_polynomial,
    compute_tutte_polynomial,
)


def graph_polynomial_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


_CYCLE_GRAPH_EXAMPLE: dict[str, Any] = {
    "graph": {
        "vertex_count": 4,
        "edges": [
            {"u": 0, "v": 1},
            {"u": 1, "v": 2},
            {"u": 2, "v": 3},
            {"u": 3, "v": 0},
        ],
    },
}

_PATH_GRAPH_EXAMPLE: dict[str, Any] = {
    "graph": {
        "vertex_count": 3,
        "edges": [
            {"u": 0, "v": 1},
            {"u": 1, "v": 2},
        ],
    },
}

_PATH_TREE_EXAMPLE: dict[str, Any] = {
    "graph": {
        "vertices": ["a", "b", "c", "d"],
        "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
    }
}


GRAPH_POLYNOMIAL_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    graph_polynomial_operation(
        "graph.polynomial.tutte.compute",
        "Compute the Tutte polynomial",
        "Compute the exact Tutte polynomial T_G(x, y) of a finite simple "
        "graph using NetworkX, with structural exponent pairs on the ordered "
        "variable axis (x, y).",
        GraphPolynomialRequest,
        SparseMultivariatePolynomial,
        compute_tutte_polynomial,
        "graph",
        "polynomial",
        "tutte",
        "exact",
        examples=(
            example(
                "cycle_graph_c4",
                "Tutte polynomial of the 4-cycle C4.",
                _CYCLE_GRAPH_EXAMPLE,
            ),
        ),
    ),
    graph_polynomial_operation(
        "graph.polynomial.chromatic.compute",
        "Compute the chromatic polynomial",
        "Compute the exact chromatic polynomial chi_G(x) of a finite simple "
        "graph using NetworkX.",
        GraphPolynomialRequest,
        GraphPolynomialResult,
        compute_chromatic_polynomial,
        "graph",
        "polynomial",
        "chromatic",
        "exact",
        examples=(
            example(
                "path_graph_p3",
                "Chromatic polynomial of the path P3.",
                _PATH_GRAPH_EXAMPLE,
            ),
        ),
    ),
    graph_polynomial_operation(
        "graph.polynomial.flow.compute",
        "Compute the flow polynomial",
        "Compute the exact nowhere-zero flow polynomial F_G(x) of a finite "
        "simple graph, derived from the Tutte polynomial.",
        GraphPolynomialRequest,
        GraphPolynomialResult,
        compute_flow_polynomial,
        "graph",
        "polynomial",
        "flow",
        "exact",
        examples=(
            example(
                "cycle_graph_c4_flow",
                "Flow polynomial of the 4-cycle C4.",
                _CYCLE_GRAPH_EXAMPLE,
            ),
        ),
    ),
    graph_polynomial_operation(
        "graph.polynomial.independence.compute",
        "Compute a tree independence polynomial",
        "Compute the exact independence polynomial whose degree-k coefficient "
        "counts independent vertex sets of cardinality k in one nonempty "
        "finite tree. Return its source-bound dense coefficients, independence "
        "number, total independent-set count, and canonical sparse "
        "RationalPolynomial in QQ[x] after a scalar preflight bounds "
        "convolution work and serialized-result size.",
        TreeIndependencePolynomialRequest,
        TreeIndependencePolynomialResult,
        compute_independence_polynomial,
        "graph",
        "polynomial",
        "independence",
        "independent-sets",
        "cardinality-count",
        "tree",
        "exact",
        examples=(
            example(
                "path_tree_p4",
                (
                    "Compute the exact independence polynomial of P4; the graph "
                    "must be a nonempty tree within this operation's bounded "
                    "convolution-work and serialized-output envelope."
                ),
                _PATH_TREE_EXAMPLE,
            ),
        ),
    ),
    graph_polynomial_operation(
        "graph.polynomial.matching.compute",
        "Compute the matching polynomial",
        "Compute the exact matching polynomial M_G(x) of a finite simple "
        "graph by the deletion recurrence on graphs with at most 16 vertices.",
        MatchingPolynomialRequest,
        GraphPolynomialResult,
        compute_matching_polynomial,
        "graph",
        "polynomial",
        "matching",
        "exact",
        examples=(
            example(
                "path_graph_p3_matching",
                "Matching polynomial of the path P3.",
                _PATH_GRAPH_EXAMPLE,
            ),
        ),
    ),
)


TOOLS = GRAPH_POLYNOMIAL_OPERATIONS

__all__ = ["TOOLS"]
