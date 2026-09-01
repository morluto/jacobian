"""Exact graph isomorphism decision operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.isomorphism._models import (
    ColoredGraphCanonicalizationRequest,
    ColoredGraphCanonicalizationResult,
    GraphIsomorphismRequest,
    GraphIsomorphismResult,
)
from jacobian.math.graphs.isomorphism._vf2_process import (
    compute_colored_graph_canonicalization,
    decide_graph_isomorphism,
)

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.isomorphism.decide.compute",
        title="Decide whether two simple graphs are isomorphic",
        description="Decide whether two simple graphs (directed or undirected) are "
        "isomorphic using a bounded NetworkX VF2 worker. Returns ISOMORPHIC "
        "with an explicit vertex mapping when an isomorphism exists, "
        "NOT_ISOMORPHIC when the exact search completes without one, or "
        "UNKNOWN when the bounded worker cannot complete. Both graphs must "
        "share the same vertex count and directedness.",
        request_type=GraphIsomorphismRequest,
        result_type=GraphIsomorphismResult,
        run=decide_graph_isomorphism,
        tags=("graph", "isomorphism", "exact"),
        examples=(
            OperationExample(
                name="path_graphs",
                description="Decide isomorphism between two path graphs.",
                input={
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
            OperationExample(
                name="nonisomorphic",
                description="Decide non-isomorphism between two distinct graphs.",
                input={
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
    MathTool(
        operation_id="graph.isomorphism.canonicalize.compute",
        title="Canonicalize a bounded colored graph under isomorphism",
        description="Return the exact canonical colored graph and an explicit "
        "source-to-canonical relabeling. Vertex and edge color names are "
        "preserved exactly; exhaustive color-class permutation work and the "
        "source-bound result size are admitted before execution.",
        request_type=ColoredGraphCanonicalizationRequest,
        result_type=ColoredGraphCanonicalizationResult,
        run=compute_colored_graph_canonicalization,
        tags=(
            "graph",
            "isomorphism",
            "canonical-form",
            "vertex-color",
            "edge-color",
            "exact",
            "bounded",
        ),
        examples=(
            OperationExample(
                name="colored_path",
                description="Canonicalize a vertex- and edge-colored four-vertex path and return its source-to-canonical relabeling; each edge must use left < right and every nonempty color axis must align with the graph's authoritative axis.",
                input={
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


__all__ = ["TOOLS"]
