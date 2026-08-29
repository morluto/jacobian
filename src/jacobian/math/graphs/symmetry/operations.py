"""Exact orbit partitions for declared finite graph symmetries."""

from __future__ import annotations

import unicodedata

from pydantic_core import PydanticCustomError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.symmetry._models import (
    _UNCOLORED,
    MAX_GRAPH_SYMMETRY_EDGES,
    MAX_GRAPH_SYMMETRY_VERTICES,
    GraphAutomorphismGenerator,
    GraphEdgeOrbit,
    GraphSymmetryOrbitResult,
    GraphSymmetryOrbitSource,
    GraphVertexOrbit,
    _require_result_output_headroom,
    _validate_automorphism_generator,
)
from jacobian.math.graphs.symmetry._orbits import declared_orbit_partitions
from jacobian.math.graphs.values import ColoredUndirectedGraph


def _admit_graph_symmetry_orbit(
    graph: ColoredUndirectedGraph,
    generators: tuple[GraphAutomorphismGenerator, ...],
) -> None:
    """Admit graph, generator, and retained-result execution bounds."""
    vertices = graph.graph.vertices
    edges = graph.graph.edges
    try:
        if len(vertices) > MAX_GRAPH_SYMMETRY_VERTICES:
            raise PydanticCustomError(
                "graph.symmetry_exceeds_max_symmetry_vertices_vertex_bound",
                f"graph symmetry exceeds the {MAX_GRAPH_SYMMETRY_VERTICES}-vertex bound",
            )
        if len(edges) > MAX_GRAPH_SYMMETRY_EDGES:
            raise PydanticCustomError(
                "graph.symmetry_exceeds_max_symmetry_edges_edge_bound",
                f"graph symmetry exceeds the {MAX_GRAPH_SYMMETRY_EDGES}-edge bound",
            )
        generator_ids = tuple(generator.generator_id for generator in generators)
        if len(set(generator_ids)) != len(generator_ids):
            raise PydanticCustomError(
                "graph.graph_symmetry_generator_identifiers_must_be_uni",
                "graph symmetry generator identifiers must be unique",
            )
        if any(
            not unicodedata.is_normalized("NFC", generator_id)
            for generator_id in generator_ids
        ):
            raise PydanticCustomError(
                "graph.symmetry_generator_identifiers_use_unicode_nfc",
                "graph symmetry generator identifiers must use Unicode NFC",
            )
        vertex_set = set(vertices)
        edge_set = set(edges)
        vertex_colors = (
            dict(zip(vertices, graph.vertex_colors, strict=True))
            if graph.vertex_colors
            else dict.fromkeys(vertices, _UNCOLORED)
        )
        edge_colors = (
            dict(zip(edges, graph.edge_colors, strict=True))
            if graph.edge_colors
            else dict.fromkeys(edges, _UNCOLORED)
        )
        for generator in generators:
            _validate_automorphism_generator(
                generator,
                vertices,
                edges,
                vertex_set,
                edge_set,
                vertex_colors,
                edge_colors,
            )
        _require_result_output_headroom(
            GraphSymmetryOrbitSource(graph=graph, generators=generators)
        )
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=("graph",), code=error.type, message=str(error)
        ) from error


def _declared_orbit_partitions(
    graph: ColoredUndirectedGraph,
    generators: tuple[GraphAutomorphismGenerator, ...],
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[tuple[tuple[str, str], ...], ...],
]:
    """Canonical vertex and edge orbit members of the declared generators."""
    vertices = tuple(sorted(graph.graph.vertices))
    edges = tuple(sorted(graph.graph.edges))
    vertex_actions = tuple(dict(generator.mapping) for generator in generators)
    return declared_orbit_partitions(vertices, edges, vertex_actions)


def graph_symmetry_orbits(
    graph: ColoredUndirectedGraph,
    generators: tuple[GraphAutomorphismGenerator, ...],
) -> GraphSymmetryOrbitResult:
    _admit_graph_symmetry_orbit(graph, generators)
    vertices = tuple(sorted(graph.graph.vertices))
    edges = tuple(sorted(graph.graph.edges))
    vertex_orbit_members, edge_orbit_members = _declared_orbit_partitions(
        graph, generators
    )
    vertex_orbits = tuple(
        GraphVertexOrbit(orbit_index=index, representative=members[0], members=members)
        for index, members in enumerate(vertex_orbit_members)
    )
    edge_orbits = tuple(
        GraphEdgeOrbit(orbit_index=index, representative=members[0], members=members)
        for index, members in enumerate(edge_orbit_members)
    )
    return GraphSymmetryOrbitResult._from_kernel(
        graph=graph,
        generators=generators,
        vertices=vertices,
        edges=edges,
        generator_ids=tuple(sorted(generator.generator_id for generator in generators)),
        vertex_orbits=vertex_orbits,
        edge_orbits=edge_orbits,
        vertex_color_mode=("DECLARED" if graph.vertex_colors else "UNCOLORED"),
        edge_color_mode="DECLARED" if graph.edge_colors else "UNCOLORED",
    )


__all__ = ["graph_symmetry_orbits"]
