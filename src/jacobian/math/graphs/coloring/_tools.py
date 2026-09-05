"""Exact graph coloring operation declarations."""

from typing import Any

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
    ListCapacityEdgeColoringRequest,
    ListCapacityEdgeColoringResult,
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


def compute_list_capacity_edge_coloring(
    request: ListCapacityEdgeColoringRequest,
) -> ListCapacityEdgeColoringResult:
    return native.list_capacity_edge_coloring(
        request.graph, request.palette, request.lists, request.capacities
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
    MathTool(
        operation_id="graph.coloring.chromatic_number.check",
        title="Check an exact chromatic-number certificate",
        description="Check an exact vertex chromatic number from both sides: a "
        "proper k-coloring proves the upper bound, while exact nonnegative "
        "rational vertex weights prove the lower bound when every independent "
        "set has weight at most one and ceil(sum(weights)) equals k. The "
        "checker exhaustively replays all 2^n subsets inside an admitted "
        "order-20 envelope, retains the graph, claim, coloring, and weights, "
        "and returns a deterministic rejection witness for invalid evidence. "
        "Coloring and weight tuples use graph.vertices order.",
        request_type=ChromaticNumberCertificateCheckRequest,
        result_type=ChromaticNumberCertificateCheckResult,
        run=compute_chromatic_number_certificate_check,
        tags=(
            "graph",
            "coloring",
            "chromatic-number",
            "fractional-clique",
            "independent-set",
            "certificate",
            "exact",
        ),
        examples=(
            OperationExample(
                name="complete_bipartite_k23_chromatic_number",
                description="Check chi(K_2,3)=2 from its bipartition coloring and exact "
                "fractional-clique weights; coloring and weights must align "
                "with graph.vertices, and the graph may have at most 20 vertices.",
                input={
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
    MathTool(
        operation_id="graph.coloring.k_colorability.decide",
        title="Decide k-colorability of a graph",
        description="Decide whether a simple undirected graph admits a proper "
        "k-coloring and return one proper coloring when it exists, using a "
        "Z3 SAT encoding bounded by the request-visible solver_conflicts "
        "budget. Colorability is claimed only on an explicit satisfying or "
        "unsatisfiable outcome. Conflict-budget exhaustion and worker failure "
        "are operational errors and establish no colorability claim.",
        request_type=KColorabilityRequest,
        result_type=KColorabilityResult,
        run=compute_k_colorability,
        tags=("graph", "coloring", "k-colorability", "exact"),
        examples=(
            OperationExample(
                name="triangle_3_colorable",
                description="Decide 3-colorability of a triangle (K3).",
                input={
                    "graph": {
                        "vertex_count": 3,
                        "edges": [[0, 1], [1, 2], [0, 2]],
                    },
                    "colors": 3,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.independent_set.maximal.decide",
        title="Decide whether a candidate set is a maximal independent set",
        description="Decide maximal independence in a bounded simple graph and return a blocking edge or addable vertex when the candidate fails.",
        request_type=MaximalIndependentSetRequest,
        result_type=MaximalIndependentSetResult,
        run=compute_maximal_independent_set_decision,
        tags=("graph", "independent-set", "maximal", "exact"),
        examples=(
            OperationExample(
                name="path_maximal_independent_set",
                description="Decide whether {0, 2} is a maximal independent set of P4.",
                input={
                    "graph": {
                        "vertex_count": 4,
                        "edges": [[0, 1], [1, 2], [2, 3]],
                    },
                    "candidate_set": [0, 2],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.edge_coloring.k_decide",
        title="Decide bounded k-edge-colorability",
        description="Given a bounded simple graph and an integer k, decide whether the "
        "graph admits a proper k-edge-coloring. The exact solver runs under "
        "the request-visible solver_conflicts budget: non-colorability "
        "requires an explicit unsatisfiable outcome. Conflict-budget exhaustion "
        "and worker failure are operational errors. A "
        "colorable decision returns one proper edge coloring as a canonical "
        "source-bound assignment value accepted by graph.edge_coloring.check.",
        request_type=EdgeKColorabilityRequest,
        result_type=EdgeKColorabilityResult,
        run=compute_edge_k_colorability,
        tags=("graph", "edge-coloring", "chromatic-index"),
        examples=(
            OperationExample(
                name="petersen_not_3_edge_colorable",
                description=(
                    "The Petersen graph has maximum degree 3 and chromatic index "
                    "4, so it is not 3-edge-colorable. The graph must be simple "
                    "with at most 20 vertices."
                ),
                input={**PETERSEN_GRAPH, "colors": 3},
            ),
        ),
    ),
    MathTool(
        operation_id="graph.edge_coloring.check",
        title="Validate a proper edge coloring",
        description="Given one source-bound edge-to-color assignment value (the graph, "
        "the palette size, and one color per edge in graph.edges order), "
        "validate that it is a proper edge coloring (no two incident edges "
        "share a color), returning a blocking edge when it is improper. "
        "Accepts the canonical value returned by graph.edge_coloring.k_decide.",
        request_type=EdgeColoringCheckRequest,
        result_type=EdgeColoringCheckResult,
        run=compute_edge_coloring_check,
        tags=("graph", "edge-coloring", "validation"),
        examples=(
            OperationExample(
                name="petersen_4_edge_colorable_check",
                description=(
                    "A 4-edge-coloring of the Petersen graph validates as proper. "
                    "The assignment must assign one color per edge in 0..colors-1."
                ),
                input={
                    "assignment": {
                        **PETERSEN_GRAPH,
                        "colors": 4,
                        "coloring": [1, 0, 1, 3, 2, 3, 0, 3, 1, 2, 0, 2, 2, 0, 1],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="graph.edge_coloring.list_capacity.assign",
        title="Assign prescribed-list edge colors under capacities",
        description="Given a simple graph, a finite palette, one allowed-color "
        "list per edge, and one upper capacity per color, find a proper edge "
        "assignment within lists and capacities or establish infeasibility. "
        "Capacities clamp to the edge count; budget exhaustion reports "
        "UNKNOWN, never infeasibility.",
        request_type=ListCapacityEdgeColoringRequest,
        result_type=ListCapacityEdgeColoringResult,
        run=compute_list_capacity_edge_coloring,
        tags=("graph", "edge-coloring", "exact"),
        examples=(
            OperationExample(
                name="path_list_capacity_feasible",
                description="Color a 2-edge path with lists [x],[x,y] and unit capacities.",
                input={
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"]],
                    },
                    "palette": ["x", "y"],
                    "lists": [
                        {"edge": ["a", "b"], "colors": ["x"]},
                        {"edge": ["b", "c"], "colors": ["x", "y"]},
                    ],
                    "capacities": [
                        {"color": "x", "capacity": 1},
                        {"color": "y", "capacity": 1},
                    ],
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
