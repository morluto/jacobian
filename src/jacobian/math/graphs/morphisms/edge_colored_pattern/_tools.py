"""Immutable declaration for coloured subgraph containment."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample

from ._models import EdgeColoredPatternRequest, EdgeColoredPatternResult
from .operations import edge_colored_subgraph_pattern_find


def _run(request: EdgeColoredPatternRequest) -> EdgeColoredPatternResult:
    return edge_colored_subgraph_pattern_find(request.pattern, request.host)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.edge_colored_subgraph_pattern.find",
        title="Find an edge-color-preserving subgraph embedding",
        description="Decide ordinary non-induced containment of one edge-colored pattern in an edge-colored host, returning one injective host-label map in the pattern vertex order. Both sources must have nonempty total edge colorings and no vertex colors. Search charges at most 10,000,000 assignments and 50,000,000 work units, returning immediately on a checked witness. Negative decisions follow complete search or a source-count obstruction; exhaustion establishes no decision.",
        request_type=EdgeColoredPatternRequest,
        result_type=EdgeColoredPatternResult,
        run=_run,
        tags=("graph", "edge-colored", "embedding", "subgraph", "exact"),
        examples=(
            OperationExample(
                name="mixed_color_path",
                description="A red-blue path occurs with both required colors preserved. Both sources need nonempty total edge colorings and empty vertex-color axes.",
                input={
                    "pattern": {
                        "graph": {
                            "vertices": ["a", "b", "c"],
                            "edges": [["a", "b"], ["b", "c"]],
                        },
                        "edge_colors": ["red", "blue"],
                    },
                    "host": {
                        "graph": {
                            "vertices": ["x", "y", "z"],
                            "edges": [["x", "y"], ["y", "z"]],
                        },
                        "edge_colors": ["red", "blue"],
                    },
                },
            ),
        ),
    ),
)
