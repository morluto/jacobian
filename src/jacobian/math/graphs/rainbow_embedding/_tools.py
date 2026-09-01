"""Rainbow embedding profile operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.rainbow_embedding._models import (
    RainbowEmbeddingRequest,
    RainbowEmbeddingResult,
)
from jacobian.math.graphs.rainbow_embedding.operations import (
    compute_rainbow_embedding_profile,
)


def compute_rainbow_embedding_op(
    request: RainbowEmbeddingRequest,
) -> RainbowEmbeddingResult:
    return compute_rainbow_embedding_profile(request.pattern, request.host)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.edge_colored.rainbow_subgraph_embedding_profile.compute",
        title="Compute the rainbow subgraph embedding profile",
        description=(
            "Given an uncoloured finite pattern graph and an edge-coloured "
            "finite host, return the complete ordered family of injective "
            "non-induced pattern embeddings whose images use pairwise "
            "distinct host-edge colours."
        ),
        request_type=RainbowEmbeddingRequest,
        result_type=RainbowEmbeddingResult,
        run=compute_rainbow_embedding_op,
        tags=("graph", "ramsey", "exact"),
        examples=(
            OperationExample(
                name="p2_in_k3",
                description="A path P2 embedded in a 3-vertex coloured complete graph.",
                input={
                    "pattern": {
                        "vertices": ["a", "b"],
                        "edges": [["a", "b"]],
                    },
                    "host": {
                        "graph": {
                            "vertices": ["0", "1", "2"],
                            "edges": [["0", "1"], ["0", "2"], ["1", "2"]],
                        },
                        "edge_colors": ["red", "blue", "green"],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
