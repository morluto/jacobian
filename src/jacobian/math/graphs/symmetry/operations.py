"""Exact orbit partitions for declared finite graph symmetries."""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from math import factorial, prod
from typing import Any

from pydantic_core import PydanticCustomError

from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_checkpoint,
)
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
from jacobian.math.groups._models import MAX_GROUP_DEGREE, PermutationGroup

MAX_FULL_AUTOMORPHISM_PERMUTATIONS = 100_000
MAX_FULL_AUTOMORPHISM_WORK = 25_600_000


def _full_graph_vertex_axis(graph: ColoredUndirectedGraph) -> tuple[str, ...]:
    return tuple(sorted(graph.graph.vertices))


def _uniform(values: tuple[str, ...]) -> bool:
    return not values or len(set(values)) == 1


def _complete_edge_colors_determined_by_vertex_classes(
    graph: ColoredUndirectedGraph,
) -> bool:
    """True when each edge color is a function of its endpoint vertex colors."""

    vertex_colors = dict(
        zip(
            graph.graph.vertices,
            graph.vertex_colors or (_UNCOLORED,) * len(graph.graph.vertices),
            strict=True,
        )
    )
    edge_colors = dict(
        zip(
            graph.graph.edges,
            graph.edge_colors or (_UNCOLORED,) * len(graph.graph.edges),
            strict=True,
        )
    )
    pair_color: dict[tuple[str, str], str] = {}
    for left, right in graph.graph.edges:
        key = tuple(sorted((vertex_colors[left], vertex_colors[right])))
        color = edge_colors[canonical_edge(left, right)]
        previous = pair_color.get(key)
        if previous is None:
            pair_color[key] = color
        elif previous != color:
            return False
    return True


@dataclass(frozen=True)
class _FullGraphAdmission:
    vertices: tuple[str, ...]
    special: tuple[tuple[tuple[int, ...], ...], int] | None
    refinement: tuple[tuple[str, int, tuple[tuple[str, str], ...]], ...] | None


def _admit_full_graph_automorphism(
    graph: ColoredUndirectedGraph,
) -> _FullGraphAdmission:
    """Admit every expansion used by the full-group kernel before searching."""

    if not isinstance(graph, ColoredUndirectedGraph):
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.automorphism.graph_type",
            message="graph must be a ColoredUndirectedGraph value",
        )
    vertex_count = len(graph.graph.vertices)
    edge_count = len(graph.graph.edges)
    if vertex_count > MAX_GROUP_DEGREE:
        raise OperationResourceAdmissionError(
            location=("graph", "vertices"),
            code="graph.automorphism.vertex_bound",
            message=(
                "full graph automorphisms expose a composable permutation group "
                f"of degree at most {MAX_GROUP_DEGREE}"
            ),
        )
    if edge_count > MAX_GRAPH_SYMMETRY_EDGES:
        raise OperationResourceAdmissionError(
            location=("graph", "edges"),
            code="graph.automorphism.edge_bound",
            message="full graph automorphism edge scans exceed the admitted carrier",
        )
    # The graph-specific special families below have compact presentations and
    # do not enumerate their groups.  Other requests are admitted against the
    # candidate-permutation envelope implied by vertex colors and local VF2
    # invariants; this bound is charged before GraphMatcher starts.
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in graph.graph.vertices}
    edge_colors = dict(
        zip(
            graph.graph.edges,
            graph.edge_colors or (_UNCOLORED,) * edge_count,
            strict=True,
        )
    )
    vertex_colors = dict(
        zip(
            graph.graph.vertices,
            graph.vertex_colors or (_UNCOLORED,) * vertex_count,
            strict=True,
        )
    )
    for left, right in graph.graph.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    vertices = _full_graph_vertex_axis(graph)
    special = _special_graph_generators(graph, vertices)
    if special is not None:
        return _FullGraphAdmission(vertices, special, None)
    classes: dict[tuple[Any, ...], list[str]] = {}
    refinement = tuple(
        (
            vertex_colors[vertex],
            len(adjacency[vertex]),
            tuple(
                sorted(
                    (
                        vertex_colors[neighbor],
                        edge_colors[canonical_edge(vertex, neighbor)],
                    )
                    for neighbor in adjacency[vertex]
                )
            ),
        )
        for vertex in vertices
    )
    for vertex, signature in zip(vertices, refinement, strict=True):
        classes.setdefault(signature, []).append(vertex)
    permutation_bound = 1
    for vertex_class in classes.values():
        permutation_bound *= factorial(len(vertex_class))
    work = permutation_bound * max(1, vertex_count + edge_count)
    if (
        permutation_bound > MAX_FULL_AUTOMORPHISM_PERMUTATIONS
        or work > MAX_FULL_AUTOMORPHISM_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("graph",),
            code="graph.automorphism.search_bound",
            message=(
                "color/refinement candidate permutations and VF2 edge-scan work "
                "exceed the admitted full-automorphism envelope"
            ),
        )
    return _FullGraphAdmission(vertices, None, refinement)


def _networkx_graph(
    graph: ColoredUndirectedGraph,
    vertices: tuple[str, ...],
    node_colors: tuple[Any, ...],
) -> Any:
    import networkx as nx

    indexed = {vertex: index for index, vertex in enumerate(vertices)}
    candidate: Any = nx.Graph()
    colors = dict(zip(vertices, node_colors, strict=True))
    candidate.add_nodes_from(
        (index, {"color": colors[vertex]}) for vertex, index in indexed.items()
    )
    edge_colors = dict(
        zip(
            graph.graph.edges,
            graph.edge_colors or (_UNCOLORED,) * len(graph.graph.edges),
            strict=True,
        )
    )
    candidate.add_edges_from(
        (
            indexed[left],
            indexed[right],
            {"color": edge_colors[(left, right)]},
        )
        for left, right in graph.graph.edges
    )
    return candidate


def _networkx_automorphisms(
    graph: ColoredUndirectedGraph,
    vertices: tuple[str, ...],
    refinement: tuple[tuple[str, int, tuple[tuple[str, str], ...]], ...],
) -> tuple[tuple[int, ...], ...]:
    """Enumerate only the admitted generic automorphisms through VF2."""

    import networkx.algorithms.isomorphism as iso

    class _CheckpointingGraphMatcher(iso.GraphMatcher):
        def semantic_feasibility(self, g1_node: Any, g2_node: Any) -> bool:
            request_checkpoint("during full graph automorphism candidate search")
            return bool(super().semantic_feasibility(g1_node, g2_node))

    source = _networkx_graph(graph, vertices, refinement)
    matcher = _CheckpointingGraphMatcher(
        source,
        source,
        node_match=iso.categorical_node_match("color", _UNCOLORED),
        edge_match=iso.categorical_edge_match("color", _UNCOLORED),
    )
    mappings: list[tuple[int, ...]] = []
    try:
        for mapping in matcher.isomorphisms_iter():
            request_checkpoint("during full graph automorphism search")
            images = tuple(mapping[index] for index in range(len(vertices)))
            mappings.append(images)
            if len(mappings) > MAX_FULL_AUTOMORPHISM_PERMUTATIONS:
                raise OperationResourceAdmissionError(
                    location=("graph",),
                    code="graph.automorphism.enumeration_bound",
                    message="generic full-automorphism enumeration exceeds its admitted bound",
                )
    except (
        OperationResourceAdmissionError,
        OperationExecutionCancelledError,
        OperationExecutionTimeoutError,
    ):
        raise
    except Exception as error:
        raise OperationBackendError(BackendFailureReason.MALFORMED_RESPONSE) from error
    return tuple(sorted(mappings))


def _special_complete_or_empty(
    graph: ColoredUndirectedGraph,
    vertices: tuple[str, ...],
    edges: set[tuple[int, int]],
) -> tuple[tuple[tuple[int, ...], ...], int] | None:
    n = len(vertices)
    complete = len(edges) == n * (n - 1) // 2
    empty = len(edges) == 0
    if not empty and not complete:
        return None
    if complete and not _complete_edge_colors_determined_by_vertex_classes(graph):
        return None
    colors = dict(
        zip(
            graph.graph.vertices,
            graph.vertex_colors or (_UNCOLORED,) * n,
            strict=True,
        )
    )
    classes: dict[str, list[int]] = {}
    for position, color in sorted(
        ((position, colors[vertex]) for position, vertex in enumerate(vertices)),
        key=lambda item: item[1],
    ):
        classes.setdefault(color, []).append(position)
    generators: list[tuple[int, ...]] = []
    order = 1
    for positions in classes.values():
        size = len(positions)
        order *= factorial(size)
        if size >= 2:
            swap = list(range(n))
            swap[positions[0]], swap[positions[1]] = positions[1], positions[0]
            generators.append(tuple(swap))
        if size >= 3:
            cycle = list(range(n))
            for source, target in zip(
                positions, positions[1:] + positions[:1], strict=True
            ):
                cycle[source] = target
            generators.append(tuple(cycle))
    return tuple(generators), order


def _indexed_adjacency(n: int, edges: set[tuple[int, int]]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {position: set() for position in range(n)}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def _position_colors(
    graph: ColoredUndirectedGraph, vertices: tuple[str, ...]
) -> tuple[tuple[str, ...], dict[frozenset[str], str]]:
    """Return vertex colors on the sorted axis and edge colors by endpoints."""

    count = len(vertices)
    vertex_color = dict(
        zip(
            graph.graph.vertices,
            graph.vertex_colors or (_UNCOLORED,) * count,
            strict=True,
        )
    )
    raw_edge_colors = graph.edge_colors or (_UNCOLORED,) * len(graph.graph.edges)
    edge_color = {
        frozenset((left, right)): color
        for (left, right), color in zip(graph.graph.edges, raw_edge_colors, strict=True)
    }
    return (
        tuple(vertex_color[vertex] for vertex in vertices),
        edge_color,
    )


def _mapped_edge_color(
    edge_color: dict[frozenset[str], str],
    vertices: tuple[str, ...],
    left: int,
    right: int,
) -> str:
    return edge_color[frozenset((vertices[left], vertices[right]))]


def _filtered_path_symmetries(
    graph: ColoredUndirectedGraph,
    vertices: tuple[str, ...],
    path: list[int],
) -> tuple[tuple[tuple[int, ...], ...], int]:
    """Filter the path reversal by the declared colors.

    The uncolored path group is the identity plus the endpoint reversal, so
    the color-preserving subgroup is exact once the reversal is checked.
    """

    count = len(vertices)
    position_colors, edge_color = _position_colors(graph, vertices)
    reflection = list(range(count))
    for source, target in zip(path, reversed(path), strict=True):
        reflection[source] = target
    for position in range(count):
        if (
            position_colors[path[position]]
            != position_colors[path[count - 1 - position]]
        ):
            return (), 1
    for position in range(count - 1):
        if _mapped_edge_color(
            edge_color, vertices, path[position], path[position + 1]
        ) != _mapped_edge_color(
            edge_color, vertices, path[count - 2 - position], path[count - 1 - position]
        ):
            return (), 1
    return (tuple(reflection),), 2


def _filtered_cycle_symmetries(
    graph: ColoredUndirectedGraph,
    vertices: tuple[str, ...],
    cycle_order: list[int],
) -> tuple[tuple[tuple[int, ...], ...], int]:
    """Filter the dihedral maps by the declared colors.

    Color-preserving rotations form a cyclic subgroup generated by the
    smallest surviving shift; color-preserving reflections (if any) form one
    coset of it. Counting both exactly yields the subgroup order alongside a
    compact generating set.
    """

    count = len(vertices)
    position_colors, edge_color = _position_colors(graph, vertices)

    def preserves_vertex(mapped: list[int]) -> bool:
        return all(
            position_colors[source] == position_colors[mapped[source]]
            for source in range(count)
        )

    def preserves_edges(mapped: list[int]) -> bool:
        return all(
            _mapped_edge_color(
                edge_color,
                vertices,
                cycle_order[position],
                cycle_order[(position + 1) % count],
            )
            == _mapped_edge_color(
                edge_color,
                vertices,
                mapped[cycle_order[position]],
                mapped[cycle_order[(position + 1) % count]],
            )
            for position in range(count)
        )

    def preserves(mapped: list[int]) -> bool:
        return preserves_vertex(mapped) and preserves_edges(mapped)

    rotation_steps: list[int] = []
    for shift in range(count):
        candidate = list(range(count))
        for position, source in enumerate(cycle_order):
            candidate[source] = cycle_order[(position + shift) % count]
        if preserves(candidate):
            rotation_steps.append(shift)
    reflection: tuple[int, ...] | None = None
    reflection_count = 0
    for offset in range(count):
        candidate = list(range(count))
        for position, source in enumerate(cycle_order):
            candidate[source] = cycle_order[(offset - position) % count]
        if preserves(candidate):
            reflection_count += 1
            if reflection is None:
                reflection = tuple(candidate)
    generators: list[tuple[int, ...]] = []
    if len(rotation_steps) > 1:
        step = min(step for step in rotation_steps if step > 0)
        generator = list(range(count))
        for position, source in enumerate(cycle_order):
            generator[source] = cycle_order[(position + step) % count]
        generators.append(tuple(generator))
    if reflection is not None:
        generators.append(reflection)
    return tuple(generators), len(rotation_steps) + reflection_count


def _special_path_or_cycle(
    graph: ColoredUndirectedGraph,
    vertices: tuple[str, ...],
    edges: set[tuple[int, int]],
) -> tuple[tuple[tuple[int, ...], ...], int] | None:
    n = len(vertices)
    if n < 3:
        return None
    colored = not _uniform(graph.vertex_colors or ()) or not _uniform(
        graph.edge_colors or ()
    )
    adjacency = _indexed_adjacency(n, edges)
    if len(edges) == n - 1 and sorted(map(len, adjacency.values())) == [
        1,
        1,
        *([2] * (n - 2)),
    ]:
        endpoints = sorted(
            position for position, neighbors in adjacency.items() if len(neighbors) == 1
        )
        path: list[int] = [endpoints[0]]
        previous = -1
        while len(path) < n:
            next_vertices = sorted(adjacency[path[-1]] - {previous})
            if len(next_vertices) != 1:
                return None
            previous, current = path[-1], next_vertices[0]
            if current in path:
                return None
            path.append(current)
        if len(set(path)) != n:
            return None
        if not colored:
            reflection = list(range(n))
            for source, target in zip(path, reversed(path), strict=True):
                reflection[source] = target
            return (tuple(reflection),), 2
        return _filtered_path_symmetries(graph, vertices, path)
    connected = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - connected:
            connected.add(neighbor)
            frontier.append(neighbor)
    if len(connected) != n or not all(
        len(neighbors) == 2 for neighbors in adjacency.values()
    ):
        return None
    cycle_order = [0]
    previous = -1
    while len(cycle_order) < n:
        choices = sorted(adjacency[cycle_order[-1]] - {previous})
        previous, current = cycle_order[-1], choices[0]
        cycle_order.append(current)
    if colored:
        return _filtered_cycle_symmetries(graph, vertices, cycle_order)
    rotation = list(range(n))
    reflection = list(range(n))
    for position, source in enumerate(cycle_order):
        rotation[source] = cycle_order[(position + 1) % n]
        reflection[source] = cycle_order[-position % n]
    return (tuple(rotation), tuple(reflection)), 2 * n


def _connected_components(
    n: int, adjacency: dict[int, set[int]]
) -> tuple[tuple[int, ...], ...]:
    components: list[tuple[int, ...]] = []
    unseen = set(range(n))
    while unseen:
        start = min(unseen)
        component = {start}
        frontier = [start]
        unseen.remove(start)
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current] & unseen:
                component.add(neighbor)
                unseen.remove(neighbor)
                frontier.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _component_class_pair_edge_profile(
    aligned: tuple[int, ...],
    vertices: tuple[str, ...],
    vertex_colors: dict[str, str],
    edge_colors: dict[tuple[str, str], str],
) -> tuple[tuple[tuple[str, str], str], ...] | None:
    """Return the edge-color map by endpoint vertex colors, or None if mixed."""

    pair_color: dict[tuple[str, str], str] = {}
    for left in aligned:
        for right in aligned:
            if left >= right:
                continue
            key = tuple(
                sorted(
                    (
                        vertex_colors[vertices[left]],
                        vertex_colors[vertices[right]],
                    )
                )
            )
            color = edge_colors[canonical_edge(vertices[left], vertices[right])]
            previous = pair_color.get(key)
            if previous is None:
                pair_color[key] = color
            elif previous != color:
                return None
    return tuple(sorted(pair_color.items()))


def _special_repeated_cliques(
    graph: ColoredUndirectedGraph,
    vertices: tuple[str, ...],
    edges: set[tuple[int, int]],
) -> tuple[tuple[tuple[int, ...], ...], int] | None:
    components = _connected_components(
        len(vertices), _indexed_adjacency(len(vertices), edges)
    )
    if len(components) <= 1:
        return None
    if any(
        sum(left in component and right in component for left, right in edges)
        != len(component) * (len(component) - 1) // 2
        for component in components
    ):
        return None
    vertex_colors = dict(
        zip(
            graph.graph.vertices,
            graph.vertex_colors or (_UNCOLORED,) * len(graph.graph.vertices),
            strict=True,
        )
    )
    edge_colors = dict(
        zip(
            graph.graph.edges,
            graph.edge_colors or (_UNCOLORED,) * len(graph.graph.edges),
            strict=True,
        )
    )
    groups: dict[tuple[object, ...], list[tuple[int, ...]]] = {}
    for component in components:
        aligned = tuple(
            sorted(
                component,
                key=lambda index: (vertex_colors[vertices[index]], vertices[index]),
            )
        )
        vertex_profile = tuple(
            sorted(Counter(vertex_colors[vertices[index]] for index in aligned).items())
        )
        edge_profile = _component_class_pair_edge_profile(
            aligned, vertices, vertex_colors, edge_colors
        )
        if edge_profile is None:
            return None
        profile = (len(aligned), vertex_profile, edge_profile)
        groups.setdefault(profile, []).append(aligned)
    generators: list[tuple[int, ...]] = []
    order = 1
    for group in groups.values():
        first = group[0]
        classes: dict[str, list[int]] = {}
        for index in first:
            classes.setdefault(vertex_colors[vertices[index]], []).append(index)
        within_order = prod(factorial(len(members)) for members in classes.values())
        for component in group:
            component_classes: dict[str, list[int]] = {}
            for index in component:
                component_classes.setdefault(vertex_colors[vertices[index]], []).append(
                    index
                )
            for members in component_classes.values():
                if len(members) >= 2:
                    swap = list(range(len(vertices)))
                    swap[members[0]], swap[members[1]] = members[1], members[0]
                    generators.append(tuple(swap))
                if len(members) >= 3:
                    cycle = list(range(len(vertices)))
                    for source, target in zip(
                        members, members[1:] + members[:1], strict=True
                    ):
                        cycle[source] = target
                    generators.append(tuple(cycle))
        for component in group[1:]:
            swap = list(range(len(vertices)))
            for left, right in zip(first, component, strict=True):
                swap[left], swap[right] = right, left
            generators.append(tuple(swap))
        order *= within_order ** len(group) * factorial(len(group))
    return tuple(generators), order


def _special_graph_generators(
    graph: ColoredUndirectedGraph, vertices: tuple[str, ...]
) -> tuple[tuple[tuple[int, ...], ...], int] | None:
    """Return compact presentations for common high-symmetry graph families."""

    if not vertices:
        return ((), 1)
    index = {vertex: position for position, vertex in enumerate(vertices)}
    edges: set[tuple[int, int]] = {
        (
            min(index[left], index[right]),
            max(index[left], index[right]),
        )
        for left, right in graph.graph.edges
    }
    return (
        _special_complete_or_empty(graph, vertices, edges)
        or _special_path_or_cycle(graph, vertices, edges)
        or _special_repeated_cliques(graph, vertices, edges)
    )


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
    """Return a compact, source-bound presentation of every graph automorphism."""

    from sympy.combinatorics import Permutation as SympyPermutation

    admission = _admit_full_graph_automorphism(graph)
    vertices = admission.vertices
    special = admission.special
    if special is None:
        if admission.refinement is None:
            raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)
        candidates = _networkx_automorphisms(graph, vertices, admission.refinement)
        identity = tuple(range(len(vertices)))
        candidate_generators = tuple(
            candidate for candidate in candidates if candidate != identity
        )
        expected_order = len(candidates)
    else:
        candidate_generators, expected_order = special

    selected: list[tuple[int, ...]] = []
    backend_group: Any | None = None
    for candidate in candidate_generators:
        request_checkpoint("during full graph automorphism generator reduction")
        permutation = SympyPermutation(list(candidate), size=len(vertices))
        if permutation.is_Identity:
            continue
        if backend_group is not None and backend_group.contains(permutation):
            continue
        selected.append(candidate)
        if len(selected) > MAX_GROUP_DEGREE:
            raise OperationResourceAdmissionError(
                location=("graph",),
                code="graph.automorphism.generator_bound",
                message=(
                    "the compact automorphism presentation exceeds the admitted "
                    f"{MAX_GROUP_DEGREE}-generator result envelope"
                ),
            )
        backend_group = __import__(
            "sympy.combinatorics", fromlist=["PermutationGroup"]
        ).PermutationGroup(
            [
                SympyPermutation(list(generator), size=len(vertices))
                for generator in selected
            ]
        )

    group_generators: tuple[tuple[int, ...], ...]
    if backend_group is None:
        generated_order = 1
        group_generators = (tuple(range(len(vertices))),)
    else:
        generated_order = int(backend_group.order())
        group_generators = tuple(selected)
    if generated_order != expected_order:
        raise OperationBackendError(BackendFailureReason.INVALID_OUTPUT)

    source_vertices = graph.graph.vertices
    generator_rows = tuple(
        GraphAutomorphismGenerator(
            generator_id=f"g{index}",
            mapping=tuple(
                (vertex, vertices[candidate[vertices.index(vertex)]])
                for vertex in source_vertices
            ),
        )
        for index, candidate in enumerate(selected)
    )
    source_actions = tuple(dict(generator.mapping) for generator in generator_rows)
    vertex_members, edge_members = declared_orbit_partitions(
        vertices,
        tuple(sorted(graph.graph.edges)),
        source_actions,
    )
    vertex_orbits = tuple(
        GraphVertexOrbit(orbit_index=index, representative=members[0], members=members)
        for index, members in enumerate(vertex_members)
    )
    edge_orbits = tuple(
        GraphEdgeOrbit(orbit_index=index, representative=members[0], members=members)
        for index, members in enumerate(edge_members)
    )
    group = PermutationGroup(degree=len(vertices), generators=group_generators)
    return FullGraphAutomorphismResult._from_kernel(
        graph=graph,
        vertices=vertices,
        edges=tuple(sorted(graph.graph.edges)),
        group=group,
        generators=generator_rows,
        order=generated_order,
        vertex_orbits=vertex_orbits,
        edge_orbits=edge_orbits,
    )


__all__ = [
    "full_graph_automorphism_group",
    "graph_symmetry_orbits",
    "verify_graph_symmetry_orbits",
]
