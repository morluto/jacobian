"""Graph morphism operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.morphisms._models import (
    CoreCheckRequest,
    CoreCheckResult,
    HomomorphismCheckRequest,
    HomomorphismCheckResult,
    HomomorphismFindRequest,
    HomomorphismFindResult,
    RetractionCheckRequest,
    RetractionCheckResult,
)
from jacobian.math.graphs.morphisms._operations import (
    compute_core_check,
    compute_homomorphism_check,
    compute_homomorphism_find,
    compute_retraction_check,
)


def _op[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "graph.homomorphism.check",
        "Check if a vertex map is a graph homomorphism",
        "Check whether a given vertex map from source to target preserves "
        "all edges of the source graph.",
        HomomorphismCheckRequest,
        HomomorphismCheckResult,
        compute_homomorphism_check,
        "graph",
        "homomorphism",
        "exact",
        examples=(
            example(
                "identity_homomorphism",
                "Check the identity map on a single edge graph.",
                {
                    "source_graph": {
                        "vertex_count": 2,
                        "edges": [[0, 1]],
                    },
                    "target_graph": {
                        "vertex_count": 2,
                        "edges": [[0, 1]],
                    },
                    "vertex_map": [0, 1],
                },
            ),
        ),
    ),
    _op(
        "graph.homomorphism.find",
        "Find a graph homomorphism if one exists",
        "Search for a homomorphism from the source graph to the target graph "
        "using backtracking. Returns whether a homomorphism exists and a "
        "witness vertex map.",
        HomomorphismFindRequest,
        HomomorphismFindResult,
        compute_homomorphism_find,
        "graph",
        "homomorphism",
        "exact",
        examples=(
            example(
                "k2_to_k2",
                "Find a homomorphism from K2 to K2.",
                {
                    "source_graph": {
                        "vertex_count": 2,
                        "edges": [[0, 1]],
                    },
                    "target_graph": {
                        "vertex_count": 2,
                        "edges": [[0, 1]],
                    },
                },
            ),
        ),
    ),
    _op(
        "graph.core.check",
        "Check if a graph is a core",
        "Check whether a graph is a core, i.e., has no non-injective "
        "endomorphism. Returns true if the graph is a core.",
        CoreCheckRequest,
        CoreCheckResult,
        compute_core_check,
        "graph",
        "core",
        "exact",
        examples=(
            example(
                "k2_is_core",
                "Check if K2 is a core.",
                {
                    "graph": {
                        "vertex_count": 2,
                        "edges": [[0, 1]],
                    },
                },
            ),
        ),
    ),
    _op(
        "graph.retraction.check",
        "Check if a retraction onto a subgraph exists",
        "Check whether there exists a homomorphism from the graph to an "
        "subgraph induced by the given vertices that fixes every vertex of "
        "the subgraph.",
        RetractionCheckRequest,
        RetractionCheckResult,
        compute_retraction_check,
        "graph",
        "retraction",
        "exact",
        examples=(
            example(
                "k3_retract_to_k2",
                "Check retraction from K3 to an edge.",
                {
                    "graph": {
                        "vertex_count": 3,
                        "edges": [[0, 1], [1, 2], [0, 2]],
                    },
                    "subgraph_vertices": [0, 1],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
