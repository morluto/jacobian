"""Exact graph polynomial operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.graphs.polynomials._models import (
    GraphPolynomialRequest,
    GraphPolynomialResult,
    MatchingPolynomialRequest,
    SparseMultivariatePolynomial,
    TreeIndependencePolynomialAdmissionError,
    TreeIndependencePolynomialRequest,
    TreeIndependencePolynomialResult,
)
from jacobian.math.graphs.polynomials.operations import (
    chromatic_polynomial,
    flow_polynomial,
    independence_polynomial_coefficients,
    matching_polynomial,
    tutte_polynomial,
)


def _run_tutte(request: GraphPolynomialRequest) -> SparseMultivariatePolynomial:
    return SparseMultivariatePolynomial(
        variables=("x", "y"),
        terms=tutte_polynomial(request.graph),
    )


def _run_chromatic(request: GraphPolynomialRequest) -> GraphPolynomialResult:
    return GraphPolynomialResult(terms=chromatic_polynomial(request.graph))


def _run_flow(request: GraphPolynomialRequest) -> GraphPolynomialResult:
    return GraphPolynomialResult(terms=flow_polynomial(request.graph))


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
    return GraphPolynomialResult(terms=matching_polynomial(request.graph))


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
    graph_polynomial_operation(
        "graph.polynomial.tutte.compute",
        "Compute the Tutte polynomial",
        "Compute the exact Tutte polynomial T_G(x, y) of a finite simple "
        "graph using NetworkX, with structural exponent pairs on the ordered "
        "variable axis (x, y).",
        GraphPolynomialRequest,
        SparseMultivariatePolynomial,
        _run_tutte,
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
        _run_chromatic,
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
        _run_flow,
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
        _run_independence,
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
        _run_matching,
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


__all__ = ["TOOLS"]
