"""Exact declared graph-symmetry operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.symmetry._models import (
    GraphSymmetryOrbitRequest,
    GraphSymmetryOrbitResult,
)
from jacobian.math.graphs.symmetry.operations import graph_symmetry_orbits


def _op[RequestT: StrictModel, ResultT: StrictModel](
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


def _compute_graph_symmetry_orbits(
    request: GraphSymmetryOrbitRequest,
) -> GraphSymmetryOrbitResult:
    """Project the wire request into the canonical native operation."""

    return graph_symmetry_orbits(request.graph, request.generators)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "graph.symmetry.generator_orbits.compute",
        "Exact declared graph-symmetry orbit partitions",
        (
            "Validate explicit color-preserving graph automorphism generators "
            "and compute the complete vertex and edge orbits of their "
            "generated subgroup. Each generator is a total vertex permutation "
            "declared as (vertex, image) pairs covering every declared vertex "
            "once in the graph's declared vertex order; generator identifiers "
            "and declared colors must already be normalized to Unicode NFC. "
            "The result retains its complete declared source action, so "
            "request validation rejects any request whose complete canonical "
            "result would exceed Jacobian's canonical output limit."
        ),
        GraphSymmetryOrbitRequest,
        GraphSymmetryOrbitResult,
        _compute_graph_symmetry_orbits,
        "graph",
        "symmetry",
        "automorphism",
        "group-action",
        "orbit",
        "compression",
        "exact",
        "bounded",
        examples=(
            example(
                "path_reflection_orbits",
                "Compute path vertex and edge orbits; the generator must be a total vertex permutation preserving colors and edges.",
                {
                    "graph": {
                        "graph": {
                            "vertices": ["a", "b", "c"],
                            "edges": [["a", "b"], ["b", "c"]],
                        },
                        "vertex_colors": ["endpoint", "middle", "endpoint"],
                    },
                    "generators": [
                        {
                            "generator_id": "reflection",
                            "mapping": [["a", "c"], ["b", "b"], ["c", "a"]],
                        }
                    ],
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
