"""Exact graph isomorphism decision operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.isomorphism._models import (
    ColoredGraphCanonicalizationRequest,
    ColoredGraphCanonicalizationResult,
    GraphIsomorphismRequest,
    GraphIsomorphismResult,
)
from jacobian.math.graphs.isomorphism._operations import (
    compute_colored_graph_canonicalization,
    decide_graph_isomorphism,
)


def graph_isomorphism_operation[
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


GRAPH_ISOMORPHISM_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    graph_isomorphism_operation(
        "graph.isomorphism.decide.compute",
        "Decide whether two simple graphs are isomorphic",
        "Decide whether two simple graphs (directed or undirected) are "
        "isomorphic using a bounded NetworkX VF2 worker. Returns ISOMORPHIC "
        "with an explicit vertex mapping when an isomorphism exists, "
        "NOT_ISOMORPHIC when the exact search completes without one, or "
        "UNKNOWN when the bounded worker cannot complete. Both graphs must "
        "share the same vertex count and directedness.",
        GraphIsomorphismRequest,
        GraphIsomorphismResult,
        decide_graph_isomorphism,
        "graph",
        "isomorphism",
        "exact",
        examples=(
            example(
                "path_graphs",
                "Decide isomorphism between two path graphs.",
                {
                    "graph_a": {
                        "vertex_count": 4,
                        "directed": False,
                        "edges": [[0, 1], [1, 2], [2, 3]],
                    },
                    "graph_b": {
                        "vertex_count": 4,
                        "directed": False,
                        "edges": [[0, 2], [2, 1], [1, 3]],
                    },
                },
            ),
            example(
                "nonisomorphic",
                "Decide non-isomorphism between two distinct graphs.",
                {
                    "graph_a": {
                        "vertex_count": 4,
                        "directed": False,
                        "edges": [[0, 1], [1, 2], [2, 3]],
                    },
                    "graph_b": {
                        "vertex_count": 4,
                        "directed": False,
                        "edges": [[0, 1], [0, 2], [0, 3]],
                    },
                },
            ),
        ),
    ),
    graph_isomorphism_operation(
        "graph.isomorphism.canonicalize.compute",
        "Canonicalize a bounded colored graph under isomorphism",
        "Return the exact canonical colored graph and an explicit "
        "source-to-canonical relabeling. Vertex and edge color names are "
        "preserved exactly; exhaustive color-class permutation work and the "
        "source-bound result size are admitted before execution.",
        ColoredGraphCanonicalizationRequest,
        ColoredGraphCanonicalizationResult,
        compute_colored_graph_canonicalization,
        "graph",
        "isomorphism",
        "canonical-form",
        "vertex-color",
        "edge-color",
        "exact",
        "bounded",
        examples=(
            example(
                "colored_path",
                "Canonicalize a vertex- and edge-colored four-vertex path and return its source-to-canonical relabeling; each edge must use left < right and every nonempty color axis must align with the graph's authoritative axis.",
                {
                    "colored_graph": {
                        "graph": {
                            "vertices": ["a", "b", "c", "d"],
                            "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
                        },
                        "vertex_colors": [
                            "endpoint",
                            "middle",
                            "middle",
                            "endpoint",
                        ],
                        "edge_colors": ["outer", "middle", "outer"],
                    }
                },
            ),
        ),
    ),
)

TOOLS = GRAPH_ISOMORPHISM_OPERATIONS

__all__ = ["TOOLS"]
