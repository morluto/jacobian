"""Declarations for divisibility-incidence graph construction."""

from jacobian.canonical import parse_canonical_integer
from jacobian.catalog._examples import example
from jacobian.math.number_theory._divisibility_graph_models import (
    DivisibilityIncidenceGraphRequest,
    DivisibilityIncidenceGraphResult,
)
from jacobian.math.number_theory._support import number_theory_operation
from jacobian.math.number_theory.operations import divisibility_incidence_graph


def compute_divisibility_incidence_graph(
    request: DivisibilityIncidenceGraphRequest,
) -> DivisibilityIncidenceGraphResult:
    """Project a wire request into the canonical graph operation."""
    return divisibility_incidence_graph(
        tuple(parse_canonical_integer(value) for value in request.left_family),
        tuple(parse_canonical_integer(value) for value in request.right_family),
    )


DIVISIBILITY_GRAPH_OPERATION = number_theory_operation(
    "number_theory.divisibility_incidence_graph.compute",
    "Build bipartite divisibility-incidence graph for two integer families",
    "Given left family L and right family R, build a bipartite simple graph joining l to r iff l divides r.",
    DivisibilityIncidenceGraphRequest,
    DivisibilityIncidenceGraphResult,
    compute_divisibility_incidence_graph,
    "number-theory",
    "graph",
    "divisibility",
    examples=(
        example(
            "div_incidence_basic",
            "Build divisibility graph for left={2,3} and right={6,12,5}; left and right families must have unique elements.",
            {"left_family": ["2", "3"], "right_family": ["6", "12", "5"]},
        ),
    ),
)

__all__ = ["DIVISIBILITY_GRAPH_OPERATION"]
