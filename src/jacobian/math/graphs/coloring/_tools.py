"""Exact graph coloring operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.coloring import operations as native
from jacobian.math.graphs.coloring._chromatic_number_models import (
    ChromaticNumberCertificateCheckRequest,
    ChromaticNumberCertificateCheckResult,
)
from jacobian.math.graphs.coloring._models import (
    EdgeColoringCheckRequest,
    EdgeColoringCheckResult,
    EdgeKColorabilityRequest,
    EdgeKColorabilityResult,
    KColorabilityRequest,
    KColorabilityResult,
    MaximalIndependentSetRequest,
    MaximalIndependentSetResult,
)


def compute_chromatic_number_certificate_check(
    request: ChromaticNumberCertificateCheckRequest,
) -> ChromaticNumberCertificateCheckResult:
    return native.chromatic_number_certificate(
        request.graph,
        request.claimed_chromatic_number,
        request.coloring,
        request.weights,
    )


def compute_k_colorability(request: KColorabilityRequest) -> KColorabilityResult:
    return native.k_colorability(
        request.graph, request.colors, request.solver_conflicts
    )


def compute_maximal_independent_set_decision(
    request: MaximalIndependentSetRequest,
) -> MaximalIndependentSetResult:
    return native.maximal_independent_set(request.graph, request.candidate_set)


def compute_edge_k_colorability(
    request: EdgeKColorabilityRequest,
) -> EdgeKColorabilityResult:
    return native.edge_k_colorability(
        request.graph, request.colors, request.solver_conflicts
    )


def compute_edge_coloring_check(
    request: EdgeColoringCheckRequest,
) -> EdgeColoringCheckResult:
    return native.edge_coloring_check(request.assignment)


def graph_coloring_operation[
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


PETERSEN_GRAPH = {
    "graph": {
        "vertices": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "edges": [
            ["0", "1"],
            ["1", "2"],
            ["2", "3"],
            ["3", "4"],
            ["0", "4"],
            ["5", "7"],
            ["7", "9"],
            ["6", "9"],
            ["6", "8"],
            ["5", "8"],
            ["0", "5"],
            ["1", "6"],
            ["2", "7"],
            ["3", "8"],
            ["4", "9"],
        ],
    }
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    graph_coloring_operation(
        "graph.coloring.chromatic_number.check",
        "Check an exact chromatic-number certificate",
        "Check an exact vertex chromatic number from both sides: a "
        "proper k-coloring proves the upper bound, while exact nonnegative "
        "rational vertex weights prove the lower bound when every independent "
        "set has weight at most one and ceil(sum(weights)) equals k. The "
        "checker exhaustively replays all 2^n subsets inside an admitted "
        "order-20 envelope, retains the graph, claim, coloring, and weights, "
        "and returns a deterministic rejection witness for invalid evidence. "
        "Coloring and weight tuples use graph.vertices order.",
        ChromaticNumberCertificateCheckRequest,
        ChromaticNumberCertificateCheckResult,
        compute_chromatic_number_certificate_check,
        "graph",
        "coloring",
        "chromatic-number",
        "fractional-clique",
        "independent-set",
        "certificate",
        "exact",
        examples=(
            example(
                "complete_bipartite_k23_chromatic_number",
                "Check chi(K_2,3)=2 from its bipartition coloring and exact "
                "fractional-clique weights; coloring and weights must align "
                "with graph.vertices, and the graph may have at most 20 vertices.",
                {
                    "graph": {
                        "vertices": ["a0", "a1", "b0", "b1", "b2"],
                        "edges": [
                            ["a0", "b0"],
                            ["a0", "b1"],
                            ["a0", "b2"],
                            ["a1", "b0"],
                            ["a1", "b1"],
                            ["a1", "b2"],
                        ],
                    },
                    "claimed_chromatic_number": 2,
                    "coloring": [0, 0, 1, 1, 1],
                    "weights": [
                        {"num": "1", "den": "2"},
                        {"num": "1", "den": "2"},
                        {"num": "1", "den": "3"},
                        {"num": "1", "den": "3"},
                        {"num": "1", "den": "3"},
                    ],
                },
            ),
        ),
    ),
    graph_coloring_operation(
        "graph.coloring.k_colorability.decide",
        "Decide k-colorability of a graph",
        "Decide whether a simple undirected graph admits a proper "
        "k-coloring and return one proper coloring when it exists, using a "
        "Z3 SAT encoding bounded by the request-visible solver_conflicts "
        "budget. Colorability is claimed only on an explicit satisfying or "
        "unsatisfiable outcome; an exhausted conflict budget yields "
        "SOLVER_BUDGET_EXCEEDED, while worker failure yields EXECUTION_FAILED; "
        "neither carries a colorability claim.",
        KColorabilityRequest,
        KColorabilityResult,
        compute_k_colorability,
        "graph",
        "coloring",
        "k-colorability",
        "exact",
        examples=(
            example(
                "triangle_3_colorable",
                "Decide 3-colorability of a triangle (K3).",
                {
                    "graph": {
                        "vertex_count": 3,
                        "edges": [[0, 1], [1, 2], [0, 2]],
                    },
                    "colors": 3,
                },
            ),
        ),
    ),
    graph_coloring_operation(
        "graph.independent_set.maximal.decide",
        "Decide whether a candidate set is a maximal independent set",
        "Decide maximal independence in a bounded simple graph and return a blocking edge or addable vertex when the candidate fails.",
        MaximalIndependentSetRequest,
        MaximalIndependentSetResult,
        compute_maximal_independent_set_decision,
        "graph",
        "independent-set",
        "maximal",
        "exact",
        examples=(
            example(
                "path_maximal_independent_set",
                "Decide whether {0, 2} is a maximal independent set of P4.",
                {
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [1, 2], [2, 3]],
                    },
                    "candidate_set": [0, 2],
                },
            ),
        ),
    ),
    graph_coloring_operation(
        "graph.edge_coloring.k_decide",
        "Decide bounded k-edge-colorability",
        "Given a bounded simple graph and an integer k, decide whether the "
        "graph admits a proper k-edge-coloring. The exact solver runs under "
        "the request-visible solver_conflicts budget: non-colorability "
        "requires an explicit unsatisfiable outcome. An exhausted conflict budget "
        "yields SOLVER_BUDGET_EXCEEDED and worker failure yields EXECUTION_FAILED; "
        "neither carries a colorability claim. A "
        "colorable decision returns one proper edge coloring as a canonical "
        "source-bound assignment value accepted by graph.edge_coloring.check.",
        EdgeKColorabilityRequest,
        EdgeKColorabilityResult,
        compute_edge_k_colorability,
        "graph",
        "edge-coloring",
        "chromatic-index",
        examples=(
            example(
                "petersen_not_3_edge_colorable",
                (
                    "The Petersen graph has maximum degree 3 and chromatic index "
                    "4, so it is not 3-edge-colorable. The graph must be simple "
                    "with at most 20 vertices."
                ),
                {**PETERSEN_GRAPH, "colors": 3},
            ),
        ),
    ),
    graph_coloring_operation(
        "graph.edge_coloring.check",
        "Validate a proper edge coloring",
        "Given one source-bound edge-to-color assignment value (the graph, "
        "the palette size, and one color per edge in graph.edges order), "
        "validate that it is a proper edge coloring (no two incident edges "
        "share a color), returning a blocking edge when it is improper. "
        "Accepts the canonical value returned by graph.edge_coloring.k_decide.",
        EdgeColoringCheckRequest,
        EdgeColoringCheckResult,
        compute_edge_coloring_check,
        "graph",
        "edge-coloring",
        "validation",
        examples=(
            example(
                "petersen_4_edge_colorable_check",
                (
                    "A 4-edge-coloring of the Petersen graph validates as proper. "
                    "The assignment must assign one color per edge in 0..colors-1."
                ),
                {
                    "assignment": {
                        **PETERSEN_GRAPH,
                        "colors": 4,
                        "coloring": [1, 0, 1, 3, 2, 3, 0, 3, 1, 2, 0, 2, 2, 0, 1],
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
