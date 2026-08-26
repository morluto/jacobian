"""Pure canonical orbit-partition kernel for declared graph symmetries."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

import networkx as nx

from jacobian.math.graphs.symmetry._edges import canonical_edge


def _orbit_components[Element: Hashable](
    elements: tuple[Element, ...],
    actions: tuple[Mapping[Element, Element], ...],
) -> tuple[tuple[Element, ...], ...]:
    union_find = nx.utils.UnionFind(elements)
    for action in actions:
        for element in elements:
            union_find.union(element, action[element])
    return tuple(tuple(members) for members in union_find.to_sets())


def declared_orbit_partitions(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    vertex_actions: tuple[Mapping[str, str], ...],
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[tuple[tuple[str, str], ...], ...],
]:
    """Return canonical vertex and edge orbits of declared generators."""

    edge_actions = tuple(
        {edge: canonical_edge(mapping[edge[0]], mapping[edge[1]]) for edge in edges}
        for mapping in vertex_actions
    )
    vertex_orbits = tuple(
        sorted(
            (
                tuple(sorted(members))
                for members in _orbit_components(vertices, vertex_actions)
            ),
            key=lambda orbit: orbit[0],
        )
    )
    edge_orbits = tuple(
        sorted(
            (
                tuple(sorted(members))
                for members in _orbit_components(edges, edge_actions)
            ),
            key=lambda orbit: orbit[0],
        )
    )
    return vertex_orbits, edge_orbits
