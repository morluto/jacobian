"""Exact orbit partitions for declared finite graph symmetries."""

from __future__ import annotations

import unicodedata
from itertools import permutations, product
from math import factorial

from pydantic_core import PydanticCustomError

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.symmetry._edges import canonical_edge
from jacobian.math.graphs.symmetry._models import (
    _UNCOLORED,
    MAX_GRAPH_SYMMETRY_EDGES,
    MAX_GRAPH_SYMMETRY_GENERATORS,
    MAX_GRAPH_SYMMETRY_VERTICES,
    FullGraphAutomorphismResult,
    GraphAutomorphismGenerator,
    GraphEdgeOrbit,
    GraphSymmetryOrbitResult,
    GraphVertexOrbit,
    _validate_automorphism_generator,
)
from jacobian.math.graphs.symmetry._orbits import declared_orbit_partitions
from jacobian.math.graphs.values import ColoredUndirectedGraph

MAX_FULL_AUTOMORPHISM_PERMUTATIONS = 100_000


def _admit_graph_symmetry_orbit(
    graph: ColoredUndirectedGraph,
    generators: tuple[GraphAutomorphismGenerator, ...],
) -> None:
    """Admit graph, generator, and retained-result execution bounds."""
    vertices = graph.graph.vertices
    edges = graph.graph.edges
    if len(generators) > MAX_GRAPH_SYMMETRY_GENERATORS:
        raise OperationResourceAdmissionError(
            location=("generators",),
            code="graph.symmetry.generator_bound",
            message="declared symmetry admits at most 64 generators",
        )
    # Preserve the old worst-case action-table envelope while admitting
    # larger graphs with fewer generators. No group elements are enumerated.
    action_entries = len(generators) * (len(vertices) + len(edges))
    # Orbit members partition the bounded vertex/edge carriers. There is at
    # most one representative and orbit record per carrier element, so the
    # result cardinality is linear even for the trivial group.
    if action_entries > 64 * (4096 + 256):
        raise OperationResourceAdmissionError(
            location=("generators",),
            code="graph.symmetry.work_bound",
            message=f"symmetry action entries={action_entries}; limit 278528",
        )
    request_checkpoint("before declared graph symmetry checking")
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
            request_checkpoint("during declared graph symmetry checking")
            _validate_automorphism_generator(
                generator,
                vertices,
                edges,
                vertex_set,
                edge_set,
                vertex_colors,
                edge_colors,
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


def verify_graph_symmetry_orbits(claim: GraphSymmetryOrbitResult) -> bool:
    """Verify generators and complete generated vertex/edge orbit partitions."""
    source = claim.source
    graph = source.graph
    vertices = graph.graph.vertices
    edges = graph.graph.edges
    try:
        _admit_graph_symmetry_orbit(graph, source.generators)
        expected_vertex_members, expected_edge_members = _declared_orbit_partitions(
            graph, source.generators
        )
    except OperationResourceAdmissionError:
        raise
    except (OperationDomainValidationError, PydanticCustomError, KeyError):
        return False

    expected_vertex_orbits = tuple(
        (members[0], members) for members in expected_vertex_members
    )
    expected_edge_orbits = tuple(
        (members[0], members) for members in expected_edge_members
    )
    return (
        claim.vertices == tuple(sorted(vertices))
        and claim.edges == tuple(sorted(edges))
        and claim.generator_ids
        == tuple(sorted(generator.generator_id for generator in source.generators))
        and claim.vertex_color_mode
        == ("DECLARED" if graph.vertex_colors else "UNCOLORED")
        and claim.edge_color_mode == ("DECLARED" if graph.edge_colors else "UNCOLORED")
        and tuple(
            (orbit.representative, orbit.members) for orbit in claim.vertex_orbits
        )
        == expected_vertex_orbits
        and tuple((orbit.representative, orbit.members) for orbit in claim.edge_orbits)
        == expected_edge_orbits
    )


def full_graph_automorphism_group(
    graph: ColoredUndirectedGraph,
) -> FullGraphAutomorphismResult:
    """Exhaust the admitted color classes and reduce the full group to generators."""

    from sympy.combinatorics import Permutation, PermutationGroup

    vertices = graph.graph.vertices
    colors = graph.vertex_colors or (_UNCOLORED,) * len(vertices)
    classes = tuple(
        tuple(index for index, value in enumerate(colors) if value == color)
        for color in sorted(set(colors))
    )
    permutation_bound = 1
    for color_class in classes:
        permutation_bound *= factorial(len(color_class))
    if permutation_bound > MAX_FULL_AUTOMORPHISM_PERMUTATIONS:
        raise OperationResourceAdmissionError(
            location=("graph", "vertex_colors"),
            code="graph.automorphism.permutation_bound",
            message="color-class permutation search exceeds its admitted bound",
        )
    edge_colors = dict(
        zip(
            graph.graph.edges,
            graph.edge_colors or (_UNCOLORED,) * len(graph.graph.edges),
            strict=True,
        )
    )
    generators: list[Permutation] = []
    generator_rows: list[GraphAutomorphismGenerator] = []
    automorphism_count = 0
    group: PermutationGroup | None = None
    class_permutations = [tuple(permutations(color_class)) for color_class in classes]
    for images_by_class in product(*class_permutations):
        images = list(range(len(vertices)))
        for color_class, class_images in zip(classes, images_by_class, strict=True):
            for source, image in zip(color_class, class_images, strict=True):
                images[source] = image
        mapping = dict(
            zip(vertices, (vertices[index] for index in images), strict=True)
        )
        mapped_edges = {
            canonical_edge(mapping[left], mapping[right]): color
            for (left, right), color in edge_colors.items()
        }
        if mapped_edges != edge_colors:
            continue
        automorphism_count += 1
        candidate = Permutation(images, size=len(vertices))
        if candidate.is_Identity or (group is not None and group.contains(candidate)):
            continue
        generators.append(candidate)
        group = PermutationGroup(generators)
        generator_rows.append(
            GraphAutomorphismGenerator(
                generator_id=f"g{len(generator_rows)}",
                mapping=tuple((vertex, mapping[vertex]) for vertex in vertices),
            )
        )
    generated_order = int(group.order()) if group is not None else 1
    if generated_order != automorphism_count:
        raise RuntimeError("reduced generators do not generate every automorphism")
    return FullGraphAutomorphismResult(
        graph=graph,
        generators=tuple(generator_rows),
        automorphism_count=automorphism_count,
        generated_group_order=generated_order,
    )


__all__ = [
    "full_graph_automorphism_group",
    "graph_symmetry_orbits",
    "verify_graph_symmetry_orbits",
]
