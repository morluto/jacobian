"""Operation declarations for k-term arithmetic-progression hypergraph construction."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.combinatorics.arithmetic_progression_hypergraph._models import (
    ArithmeticProgressionHypergraphRequest,
    ArithmeticProgressionHypergraphResult,
)
from jacobian.math.combinatorics.arithmetic_progression_hypergraph.operations import (
    construct_arithmetic_progression_hypergraph,
)


def _construct(
    request: ArithmeticProgressionHypergraphRequest,
) -> ArithmeticProgressionHypergraphResult:
    return construct_arithmetic_progression_hypergraph(
        request.lower, request.upper, request.k
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="hypergraph.arithmetic_progression.construct",
        title="Construct the k-term arithmetic-progression hypergraph of an interval",
        description=(
            "Given an inclusive finite integer interval [L,U] and arity k >= 3, "
            "return the canonical finite k-uniform hypergraph whose vertices are "
            "the integers in [L,U] and whose edges are precisely the increasing "
            "k-term arithmetic progressions contained in that interval. "
            "Each edge is labelled (a,d) with first term a and common difference d."
        ),
        request_type=ArithmeticProgressionHypergraphRequest,
        result_type=ArithmeticProgressionHypergraphResult,
        run=_construct,
        tags=(
            "combinatorics",
            "hypergraph",
            "arithmetic-progression",
            "exact",
            "bounded",
        ),
        discovery_terms=(
            "arithmetic progression hypergraph",
            "AP-free",
            "van der Waerden",
            "Erdos",
        ),
        examples=(
            example(
                "small_interval_k3",
                "Construct the 3-uniform AP hypergraph of [1,5]: 3 edges.",
                {"lower": 1, "upper": 5, "k": 3},
            ),
            example(
                "empty_interval_k3",
                "Construct the 3-uniform AP hypergraph of [1,2]: 0 edges.",
                {"lower": 1, "upper": 2, "k": 3},
            ),
            example(
                "singleton_k4",
                "Construct the 4-uniform AP hypergraph of [0,3]: 1 edge.",
                {"lower": 0, "upper": 3, "k": 4},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
