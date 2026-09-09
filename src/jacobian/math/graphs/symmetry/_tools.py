"""Exact declared graph-symmetry operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.graphs.symmetry._models import (
    FullGraphAutomorphismRequest,
    FullGraphAutomorphismResult,
    GraphSymmetryOrbitRequest,
    GraphSymmetryOrbitResult,
)
from jacobian.math.graphs.symmetry.operations import (
    full_graph_automorphism_group,
    graph_symmetry_orbits,
)


def _compute_graph_symmetry_orbits(
    request: GraphSymmetryOrbitRequest,
) -> GraphSymmetryOrbitResult:
    """Project the wire request into the canonical native operation."""

    return graph_symmetry_orbits(request.graph, request.generators)


def _compute_full_automorphisms(
    request: FullGraphAutomorphismRequest,
) -> FullGraphAutomorphismResult:
    return full_graph_automorphism_group(request.graph)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="graph.symmetry.generator_orbits.compute",
        title="Exact declared graph-symmetry orbit partitions",
        description=(
            "Validate explicit color-preserving graph automorphism generators "
            "and compute the complete vertex and edge orbits of their "
            "generated subgroup. Each generator is a total vertex permutation "
            "declared as (vertex, image) pairs covering every declared vertex "
            "once in the graph's declared vertex order; generator identifiers "
            "and declared colors must already be normalized to Unicode NFC. "
            "The result retains its complete declared source action and returns "
            "the bounded vertex and edge orbit partitions directly. Admits "
            "the 256-vertex graph carrier, at most 64 generators, 278,528 "
            "generator actions on vertices/edges, and at most one orbit record "
            "per vertex or edge."
        ),
        request_type=GraphSymmetryOrbitRequest,
        result_type=GraphSymmetryOrbitResult,
        run=_compute_graph_symmetry_orbits,
        tags=(
            "graph",
            "symmetry",
            "automorphism",
            "group-action",
            "orbit",
            "compression",
            "exact",
            "bounded",
        ),
        examples=(
            OperationExample(
                name="path_reflection_orbits",
                description="Compute path vertex and edge orbits; the generator must be a total vertex permutation preserving colors and edges.",
                input={
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
    MathTool(
        operation_id="graph.automorphism_group.color_preserving.compute",
        title="Compute the full color-preserving graph automorphism group",
        description="Exhaust the admitted color-class permutations and return deterministic generators whose Schreier-Sims order equals the full automorphism count.",
        request_type=FullGraphAutomorphismRequest,
        result_type=FullGraphAutomorphismResult,
        run=_compute_full_automorphisms,
        tags=("graph", "automorphism", "permutation-group", "exact"),
        examples=(
            OperationExample(
                name="path_reflection",
                description="Compute the order-two automorphism group of a path.",
                input={
                    "graph": {
                        "graph": {
                            "vertices": ["a", "b", "c"],
                            "edges": [["a", "b"], ["b", "c"]],
                        }
                    }
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
