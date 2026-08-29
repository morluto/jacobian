"""k-regular subgraph operation declarations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.regular_subgraph._models import (
    RegularSubgraphRequest,
    RegularSubgraphResult,
)
from jacobian.math.graphs.regular_subgraph.operations import (
    find_k_regular_subgraph,
)


def _find(request: RegularSubgraphRequest) -> RegularSubgraphResult:
    return find_k_regular_subgraph(request.graph, request.k)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.k_regular_subgraph.find",
        title="Find a k-regular subgraph",
        description=(
            "Find a nonempty k-regular subgraph of a simple undirected "
            "graph: a vertex set and edge set where every used vertex has "
            "degree exactly k. Returns a witness or found=false. "
            "Exhaustive edge-subset enumeration."
        ),
        request_type=RegularSubgraphRequest,
        result_type=RegularSubgraphResult,
        run=_find,
        tags=(
            "graph",
            "regular",
            "subgraph",
            "exact",
            "bounded",
        ),
        examples=(
            example(
                "c4_is_2_regular",
                "C4 contains a 2-regular subgraph (itself).",
                {
                    "graph": {
                        "vertices": ["a", "b", "c", "d"],
                        "edges": [
                            ["a", "b"],
                            ["b", "c"],
                            ["c", "d"],
                            ["a", "d"],
                        ],
                    },
                    "k": 2,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
