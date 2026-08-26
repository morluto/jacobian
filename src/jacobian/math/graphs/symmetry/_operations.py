"""Exact orbit partitions for declared finite graph symmetries."""

from __future__ import annotations

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.symmetry._models import (
    GraphEdgeOrbit,
    GraphSymmetryOrbitRequest,
    GraphSymmetryOrbitResult,
    GraphVertexOrbit,
)
from jacobian.math.graphs.symmetry._orbits import declared_orbit_partitions


def _declared_orbit_partitions(
    request: GraphSymmetryOrbitRequest,
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[tuple[tuple[str, str], ...], ...],
]:
    """Canonical vertex and edge orbit members of the declared generators."""
    vertices = tuple(sorted(request.graph.graph.vertices))
    edges = tuple(sorted(request.graph.graph.edges))
    vertex_actions = tuple(dict(generator.mapping) for generator in request.generators)
    return declared_orbit_partitions(vertices, edges, vertex_actions)


def _generator_orbits(
    request: GraphSymmetryOrbitRequest,
) -> GraphSymmetryOrbitResult:
    vertices = tuple(sorted(request.graph.graph.vertices))
    edges = tuple(sorted(request.graph.graph.edges))
    vertex_orbit_members, edge_orbit_members = _declared_orbit_partitions(request)
    vertex_orbits = tuple(
        GraphVertexOrbit(
            orbit_index=index,
            representative=members[0],
            members=members,
        )
        for index, members in enumerate(vertex_orbit_members)
    )
    edge_orbits = tuple(
        GraphEdgeOrbit(
            orbit_index=index,
            representative=members[0],
            members=members,
        )
        for index, members in enumerate(edge_orbit_members)
    )
    return GraphSymmetryOrbitResult._from_kernel(
        source=request,
        vertices=vertices,
        edges=edges,
        generator_ids=tuple(
            sorted(generator.generator_id for generator in request.generators)
        ),
        vertex_orbits=vertex_orbits,
        edge_orbits=edge_orbits,
        vertex_color_mode=("DECLARED" if request.graph.vertex_colors else "UNCOLORED"),
        edge_color_mode="DECLARED" if request.graph.edge_colors else "UNCOLORED",
    )


def verify_graph_symmetry_orbit_result(result: GraphSymmetryOrbitResult) -> bool:
    """Independently verify a bounded source-bound orbit-partition claim."""

    expected_vertex_members, expected_edge_members = _declared_orbit_partitions(
        result.source
    )
    return (
        tuple(orbit.members for orbit in result.vertex_orbits)
        == expected_vertex_members
        and tuple(orbit.members for orbit in result.edge_orbits)
        == expected_edge_members
    )


GRAPH_SYMMETRY_OPERATIONS: MathTools = (
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
            "The result retains its complete declared source request, so "
            "request validation rejects any request whose complete canonical "
            "result would exceed Jacobian's canonical output limit."
        ),
        request_type=GraphSymmetryOrbitRequest,
        result_type=GraphSymmetryOrbitResult,
        run=_generator_orbits,
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
            example(
                "path_reflection_orbits",
                "Compute path vertex and edge orbits; the generator must be a total vertex permutation preserving colors and edges.",
                {
                    "graph": {
                        "graph": {
                            "vertices": ["a", "b", "c"],
                            "edges": [
                                ["a", "b"],
                                ["b", "c"],
                            ],
                        },
                        "vertex_colors": ["endpoint", "middle", "endpoint"],
                    },
                    "generators": [
                        {
                            "generator_id": "reflection",
                            "mapping": [
                                ["a", "c"],
                                ["b", "b"],
                                ["c", "a"],
                            ],
                        }
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["GRAPH_SYMMETRY_OPERATIONS"]
