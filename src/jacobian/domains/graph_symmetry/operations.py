"""Exact orbit partitions for declared finite graph symmetries."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

from jacobian.contracts.graph_symmetry import (
    GraphEdgeOrbit,
    GraphSymmetryOrbitRequest,
    GraphSymmetryOrbitResult,
    GraphVertexOrbit,
)
from jacobian.domains._examples import example
from jacobian.operations import ComputedOperation, ComputedSuccess


def _canonical_edge(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _orbit_components[Element: Hashable](
    elements: tuple[Element, ...],
    actions: tuple[Mapping[Element, Element], ...],
) -> tuple[tuple[Element, ...], ...]:
    parent = {element: element for element in elements}

    def find(element: Element) -> Element:
        root = element
        while parent[root] != root:
            root = parent[root]
        while parent[element] != element:
            next_element = parent[element]
            parent[element] = root
            element = next_element
        return root

    def union(left: Element, right: Element) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        parent[right_root] = left_root

    for action in actions:
        for element in elements:
            union(element, action[element])

    members_by_root: dict[Element, list[Element]] = {}
    for element in elements:
        members_by_root.setdefault(find(element), []).append(element)
    return tuple(tuple(members) for members in members_by_root.values())


def _vertex_orbits(
    elements: tuple[str, ...],
    actions: tuple[Mapping[str, str], ...],
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        sorted(
            (
                tuple(sorted(members))
                for members in _orbit_components(elements, actions)
            ),
            key=lambda orbit: orbit[0],
        )
    )


def _edge_orbits(
    elements: tuple[tuple[str, str], ...],
    actions: tuple[Mapping[tuple[str, str], tuple[str, str]], ...],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    return tuple(
        sorted(
            (
                tuple(sorted(members))
                for members in _orbit_components(elements, actions)
            ),
            key=lambda orbit: orbit[0],
        )
    )


def _generator_orbits(
    request: GraphSymmetryOrbitRequest,
) -> ComputedSuccess[GraphSymmetryOrbitResult]:
    vertices = tuple(sorted(request.graph.vertices))
    edges = tuple(sorted(request.graph.edges))
    vertex_actions = tuple(generator.mapping for generator in request.generators)
    edge_actions = tuple(
        {edge: _canonical_edge(mapping[edge[0]], mapping[edge[1]]) for edge in edges}
        for mapping in vertex_actions
    )
    vertex_orbit_members = _vertex_orbits(vertices, vertex_actions)
    edge_orbit_members = _edge_orbits(edges, edge_actions)
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
    return ComputedSuccess(
        GraphSymmetryOrbitResult(
            vertices=vertices,
            edges=edges,
            generator_ids=tuple(
                sorted(generator.generator_id for generator in request.generators)
            ),
            generator_count=len(request.generators),
            vertex_orbits=vertex_orbits,
            edge_orbits=edge_orbits,
            vertex_orbit_count=len(vertex_orbits),
            edge_orbit_count=len(edge_orbits),
            vertex_color_mode=("DECLARED" if request.vertex_colors else "UNCOLORED"),
            edge_color_mode="DECLARED" if request.edge_colors else "UNCOLORED",
        )
    )


GRAPH_SYMMETRY_CAPABILITIES = (
    ComputedOperation(
        capability_id="graph.symmetry.generator_orbits.compute",
        title="Exact declared graph-symmetry orbit partitions",
        description=(
            "Validate explicit color-preserving graph automorphism generators and "
            "compute the complete vertex and edge orbits of their generated subgroup."
        ),
        request_model=GraphSymmetryOrbitRequest,
        result_model=GraphSymmetryOrbitResult,
        implementation=_generator_orbits,
        relation_id="graph.symmetry.generator_orbits.relation",
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
        invocation_examples=(
            example(
                "cycle_rotation_orbits",
                "Compute vertex and edge orbits of one declared quarter-turn of C4.",
                {
                    "graph": {
                        "vertices": ["a", "b", "c", "d"],
                        "edges": [
                            ["a", "b"],
                            ["a", "d"],
                            ["b", "c"],
                            ["c", "d"],
                        ],
                    },
                    "generators": [
                        {
                            "generator_id": "quarter_turn",
                            "mapping": {
                                "a": "b",
                                "b": "c",
                                "c": "d",
                                "d": "a",
                            },
                        }
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["GRAPH_SYMMETRY_CAPABILITIES"]
