"""Rainbow embedding profile operation declarations."""

from collections.abc import Callable

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def reb_action[RequestT: StrictModel, ResultT: StrictModel](
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


TOOLS: MathTools = (
    reb_action(
        "graph.edge_colored.rainbow_subgraph_embedding_profile.compute",
        "Compute the rainbow subgraph embedding profile",
        (
            "Given an uncoloured finite pattern graph and an edge-coloured "
            "finite host, return the complete ordered family of injective "
            "non-induced pattern embeddings whose images use pairwise "
            "distinct host-edge colours."
        ),
        RainbowEmbeddingRequest,
        RainbowEmbeddingResult,
        compute_rainbow_embedding_op,
        "graph",
        "ramsey",
        "exact",
        examples=(
            example(
                "p2_in_k3",
                "A path P2 embedded in a 3-vertex coloured complete graph.",
                {
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
