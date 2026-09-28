"""Exact quotient cell structure for bounded Bieberbach polygons."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from fractions import Fraction
from typing import NoReturn

from pydantic import ValidationError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.crystallographic.extensions._models import (
    BieberbachFaceOrbitComplex,
    BieberbachFaceOrbitMap,
    BieberbachGroupRingBoundaryEntry,
    CrystallographicAffineRealization,
    CrystallographicFundamentalDomainResult,
    CrystallographicPolytopePairing,
    FiniteLatticeExtension,
)
from jacobian.math.geometry.crystallographic.extensions.operations import (
    _extension_product,
    _inverse_extension_element,
    check_crystallographic_fundamental_domain,
    decide_extension_torsion,
)
from jacobian.math.topology.chain_complexes.values import (
    ChainComplexValue,
    CoefficientRing,
)

MAX_FACE_ORBIT_POLYGON_VERTICES = 32
MAX_FACE_ORBIT_POLYGON_FACETS = 32
MAX_FACE_ORBIT_RESULT_BYTES = 20_000_000

_Element = tuple[tuple[int, ...], int]


def _domain(
    reason: str, message: str, location: tuple[str | int, ...]
) -> NoReturn:
    raise OperationDomainValidationError(
        location=location,
        code=f"crystallographic.face_orbits.{reason}",
        message=message,
    )


def _resource(reason: str, message: str) -> NoReturn:
    raise OperationResourceAdmissionError(
        location=("source",),
        code=f"crystallographic.face_orbits.{reason}",
        message=message,
    )


def _multiply(source: FiniteLatticeExtension, left: _Element, right: _Element) -> _Element:
    return _extension_product(source, left, right)


def _inverse(source: FiniteLatticeExtension, value: _Element) -> _Element:
    return _inverse_extension_element(source, value)


def _facet_cycle(checked: CrystallographicFundamentalDomainResult) -> tuple[int, ...]:
    """Recover a polygon cycle from its exact edge endpoint incidence graph."""
    profile = checked.source.facet_profile
    edge_vertices = tuple(
        tuple(facet.source_vertex_indices) for facet in profile.facets
    )
    if any(len(vertices) != 2 for vertices in edge_vertices):
        _domain(
            "polygon_edge_shape",
            "every two-dimensional polygon facet must have exactly two endpoint vertices",
            ("source", "facet_profile"),
        )
    adjacency: dict[int, list[int]] = {
        index: [] for index in range(len(profile.vertices))
    }
    edge_for_pair: dict[frozenset[int], int] = {}
    for facet_index, endpoints in enumerate(edge_vertices):
        left, right = endpoints
        adjacency[left].append(right)
        adjacency[right].append(left)
        edge_for_pair[frozenset(endpoints)] = facet_index
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        _domain(
            "polygon_vertex_incidence",
            "polygon vertices must form a single 2-regular boundary cycle",
            ("source", "facet_profile"),
        )
    start = min(adjacency)
    cycle = [start]
    previous: int | None = None
    current = start
    while True:
        candidates = sorted(n for n in adjacency[current] if n != previous)
        next_vertex = candidates[0]
        if next_vertex == start:
            break
        if next_vertex in cycle:
            _domain(
                "polygon_boundary_cycle",
                "facet incidences do not form one simple polygon cycle",
                ("source", "facet_profile"),
            )
        cycle.append(next_vertex)
        previous, current = current, next_vertex
    if len(cycle) != len(adjacency):
        _domain(
            "polygon_boundary_connected",
            "polygon facet incidences must include every vertex in one cycle",
            ("source", "facet_profile"),
        )
    coords = tuple(
        tuple(value.as_fraction() for value in vertex.coordinates)
        for vertex in profile.vertices
    )
    area2 = sum(
        coords[cycle[i]][0] * coords[cycle[(i + 1) % len(cycle)]][1]
        - coords[cycle[(i + 1) % len(cycle)]][0] * coords[cycle[i]][1]
        for i in range(len(cycle))
    )
    if area2 == 0:
        _domain("polygon_area", "fundamental polygon has zero signed area", ("source",))
    if area2 < 0:
        cycle = [cycle[0], *reversed(cycle[1:])]
    return tuple(cycle)


def _element_for_pairing(pairing: CrystallographicPolytopePairing) -> _Element:
    return (tuple(pairing.lattice_translation), pairing.holonomy_element)


def _apply_element(
    realization: CrystallographicAffineRealization,
    element: _Element,
    point: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    translation, holonomy = element
    section_map = realization.section_maps[holonomy]
    return tuple(
        sum(
            section_map.linear_part[row][column] * point[column]
            for column in range(len(point))
        )
        + section_map.section_shift[row].as_fraction()
        + translation[row]
        for row in range(len(point))
    )


def _group_labels_to_vertex_representatives(
    source: FiniteLatticeExtension,
    vertex_count: int,
    maps: tuple[BieberbachFaceOrbitMap, ...],
) -> tuple[tuple[int, ...], tuple[_Element, ...]]:
    """Return canonical vertex orbit labels and deck maps to each root."""
    graph: list[list[tuple[int, _Element]]] = [[] for _ in range(vertex_count)]
    for item in maps:
        element = (tuple(item.lattice_translation), item.holonomy_element)
        inverse = _inverse(source, element)
        graph[item.source_vertex_index].append((item.target_vertex_index, element))
        graph[item.target_vertex_index].append((item.source_vertex_index, inverse))
    orbit_ids = [-1] * vertex_count
    to_root: list[_Element | None] = [None] * vertex_count
    identity: _Element = ((0,) * len(source.action_matrices[0]), 0)
    orbit_id = 0
    for root in range(vertex_count):
        if orbit_ids[root] >= 0:
            continue
        orbit_ids[root] = orbit_id
        to_root[root] = identity
        queue = deque((root,))
        while queue:
            current = queue.popleft()
            current_to_root = to_root[current]
            assert current_to_root is not None
            for neighbor, current_to_neighbor in graph[current]:
                if orbit_ids[neighbor] >= 0:
                    if orbit_ids[neighbor] != orbit_id:
                        raise ArithmeticError("vertex orbit graph crossed components")
                    continue
                # current_to_neighbor maps current -> neighbor. Compose with
                # current_to_root (neighbor -> root) on the left.
                neighbor_to_root = _multiply(source, current_to_root, _inverse(source, current_to_neighbor))
                orbit_ids[neighbor] = orbit_id
                to_root[neighbor] = neighbor_to_root
                queue.append(neighbor)
        orbit_id += 1
    return tuple(orbit_ids), tuple(value for value in to_root if value is not None)


def _one_skeleton_boundary(
    edge_orbits: list[int],
    boundary_direction: dict[int, tuple[int, int]],
    vertex_row: tuple[int, ...],
    vertex_to_root: tuple[_Element, ...],
) -> tuple[list[list[int]], list[BieberbachGroupRingBoundaryEntry]]:
    d1 = [[0 for _ in edge_orbits] for _ in set(vertex_row)]
    entries: list[BieberbachGroupRingBoundaryEntry] = []
    for column, representative in enumerate(edge_orbits):
        ordered = boundary_direction.get(representative)
        if ordered is None:
            raise ArithmeticError("edge does not occur once in the polygon cycle")
        start, end = ordered
        d1[vertex_row[start]][column] -= 1
        d1[vertex_row[end]][column] += 1
        for vertex, coefficient in ((start, -1), (end, 1)):
            label = vertex_to_root[vertex]
            entries.append(
                BieberbachGroupRingBoundaryEntry(
                    source_cell_index=column,
                    target_cell_index=vertex_row[vertex],
                    coefficient=coefficient,
                    incidence_index=vertex,
                    lattice_translation=label[0],
                    holonomy_element=label[1],
                )
            )
    return d1, entries


def _two_cell_boundary(
    boundary_facets: tuple[int, ...],
    boundary_direction: dict[int, tuple[int, int]],
    pairing_by_source: Mapping[int, CrystallographicPolytopePairing],
    edge_orbits: list[int],
    edge_row: dict[int, int],
    coordinates: tuple[tuple[Fraction, ...], ...],
    realization: CrystallographicAffineRealization,
) -> tuple[list[list[int]], list[BieberbachGroupRingBoundaryEntry]]:
    d2 = [[0] for _ in edge_orbits]
    entries: list[BieberbachGroupRingBoundaryEntry] = []
    for facet_index in boundary_facets:
        pairing = pairing_by_source[facet_index]
        representative = min(facet_index, pairing.target_facet_index)
        column = edge_row[representative]
        source_endpoints = boundary_direction[facet_index]
        target_endpoints = boundary_direction[representative]
        element = (
            _element_for_pairing(pairing)
            if facet_index != representative
            else ((0, 0), 0)
        )
        mapped = tuple(
            next(
                vertex
                for vertex in target_endpoints
                if _apply_element(realization, element, coordinates[source_vertex])
                == coordinates[vertex]
            )
            for source_vertex in source_endpoints
        )
        coefficient = 1 if mapped == target_endpoints else -1
        d2[column][0] += coefficient
        entries.append(
            BieberbachGroupRingBoundaryEntry(
                source_cell_index=0,
                target_cell_index=column,
                coefficient=coefficient,
                incidence_index=facet_index,
                lattice_translation=element[0],
                holonomy_element=element[1],
            )
        )
    return d2, entries


def _require_group_ring_boundary_zero(
    source: FiniteLatticeExtension,
    d1_entries: list[BieberbachGroupRingBoundaryEntry],
    d2_entries: list[BieberbachGroupRingBoundaryEntry],
    edge_orbits: list[int],
    profile_edges: dict[int, frozenset[int]],
    boundary_direction: dict[int, tuple[int, int]],
    coordinates: tuple[tuple[Fraction, ...], ...],
    realization: CrystallographicAffineRealization,
) -> None:
    d1_by_endpoint = {
        (entry.source_cell_index, entry.incidence_index): entry
        for entry in d1_entries
    }
    terms: dict[tuple[int, _Element], int] = {}
    for entry in d2_entries:
        facet_index = entry.incidence_index
        representative = edge_orbits[entry.target_cell_index]
        edge_element = (tuple(entry.lattice_translation), entry.holonomy_element)
        for polygon_vertex in boundary_direction[facet_index]:
            representative_vertex = next(
                vertex
                for vertex in profile_edges[representative]
                if _apply_element(
                    realization, edge_element, coordinates[polygon_vertex]
                )
                == coordinates[vertex]
            )
            endpoint = d1_by_endpoint[(entry.target_cell_index, representative_vertex)]
            endpoint_element = (
                tuple(endpoint.lattice_translation), endpoint.holonomy_element
            )
            composed = _multiply(source, endpoint_element, edge_element)
            key = (endpoint.target_cell_index, composed)
            terms[key] = terms.get(key, 0) + entry.coefficient * endpoint.coefficient
    if any(terms.values()):
        _domain(
            "group_ring_boundary_not_square_zero",
            "group-labelled face incidences fail the group-ring cellular boundary relation",
            ("source", "pairings"),
        )


def quotient_face_orbit_complex(
    result: CrystallographicFundamentalDomainResult,
) -> BieberbachFaceOrbitComplex:
    """Construct the augmented quotient cellular complex of a checked polygon.

    Rechecks the fundamental-domain source before using its positive claim.
    In dimension two, each codimension-one face is an edge, and the verified
    directed side maps induce all endpoint orbit maps. This returns ordinary
    integral quotient chains and labelled polygon incidences, not a free
    ZGamma resolution.
    """
    try:
        checked = CrystallographicFundamentalDomainResult.model_validate(
            result.model_dump(mode="python", warnings=False), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("source",),
            code="crystallographic.face_orbits.source_shape",
            message="source does not satisfy the fundamental-domain result contract",
        ) from exc
    pairing_result = checked.source
    rank = len(pairing_result.affine_realization.source.action_matrices[0])
    vertex_count = len(pairing_result.facet_profile.vertices)
    facet_count = len(pairing_result.facet_profile.facets)
    if rank != 2:
        _resource("dimension_bound", "quotient face-orbit chains currently support dimension two")
    if vertex_count > MAX_FACE_ORBIT_POLYGON_VERTICES or facet_count > MAX_FACE_ORBIT_POLYGON_FACETS:
        _resource("polygon_size_bound", "polygon exceeds the 32-vertex/facet exact face-orbit envelope")
    source_bytes = len(checked.model_dump_json().encode("utf-8"))
    predicted_bytes = source_bytes + 256 * vertex_count + 512 * facet_count + 64_000
    if predicted_bytes > MAX_FACE_ORBIT_RESULT_BYTES:
        _resource("result_bound", "face-orbit complex exceeds its result byte envelope")
    recomputed = check_crystallographic_fundamental_domain(pairing_result)
    if not recomputed.is_fundamental_domain or recomputed != checked:
        _domain("source_not_fundamental", "source must be a freshly checked fundamental polygon", ("source",))
    torsion = decide_extension_torsion(pairing_result.affine_realization.source)
    if not torsion.torsion_free:
        _domain(
            "source_has_torsion",
            "quotient cellular chains require a torsion-free crystallographic group",
            ("source", "affine_realization", "source"),
        )

    cycle = _facet_cycle(checked)
    source_extension = pairing_result.affine_realization.source
    coordinates = tuple(
        tuple(value.as_fraction() for value in vertex.coordinates)
        for vertex in pairing_result.facet_profile.vertices
    )
    pairing_by_source = {p.source_facet_index: p for p in pairing_result.pairings}
    profile_edges = {
        index: frozenset(facet.source_vertex_indices)
        for index, facet in enumerate(pairing_result.facet_profile.facets)
    }
    facet_by_edge = {vertices: index for index, vertices in profile_edges.items()}
    boundary_facets = tuple(
        facet_by_edge[frozenset((cycle[index], cycle[(index + 1) % len(cycle)]))]
        for index in range(len(cycle))
    )
    boundary_direction = {
        boundary_facets[index]: (cycle[index], cycle[(index + 1) % len(cycle)])
        for index in range(len(cycle))
    }

    maps: list[BieberbachFaceOrbitMap] = []
    for pairing in pairing_result.pairings:
        source_facet = pairing_result.facet_profile.facets[pairing.source_facet_index]
        target_vertices = set(
            pairing_result.facet_profile.facets[pairing.target_facet_index].source_vertex_indices
        )
        for source_vertex in source_facet.source_vertex_indices:
            image = _apply_element(
                pairing_result.affine_realization,
                _element_for_pairing(pairing),
                coordinates[source_vertex],
            )
            target_vertex = next(
                (idx for idx in target_vertices if coordinates[idx] == image), None
            )
            if target_vertex is None:
                _domain("endpoint_map", "side-pairing affine map misses an endpoint", ("source", "pairings", pairing.source_facet_index))
            maps.append(
                BieberbachFaceOrbitMap(
                    source_facet_index=pairing.source_facet_index,
                    source_vertex_index=source_vertex,
                    target_facet_index=pairing.target_facet_index,
                    target_vertex_index=target_vertex,
                    lattice_translation=pairing.lattice_translation,
                    holonomy_element=pairing.holonomy_element,
                )
            )
    orbit_ids, vertex_to_root = _group_labels_to_vertex_representatives(
        source_extension, vertex_count, tuple(maps)
    )
    orbit_members: dict[int, list[int]] = {}
    for vertex, orbit in enumerate(orbit_ids):
        orbit_members.setdefault(orbit, []).append(vertex)
    orbit_roots = tuple(min(vertices) for vertices in orbit_members.values())
    root_to_row = {root: row for row, root in enumerate(orbit_roots)}
    vertex_row = tuple(root_to_row[min(orbit_members[orbit])] for orbit in orbit_ids)

    edge_orbits: list[int] = []
    for facet_index, pairing in enumerate(pairing_result.pairings):
        if facet_index < pairing.target_facet_index:
            edge_orbits.append(facet_index)
    edge_orbits.sort()
    edge_row = {facet: row for row, facet in enumerate(edge_orbits)}
    d1, d1_group = _one_skeleton_boundary(
        edge_orbits, boundary_direction, vertex_row, vertex_to_root
    )
    d2, d2_group = _two_cell_boundary(
        boundary_facets,
        boundary_direction,
        pairing_by_source,
        edge_orbits,
        edge_row,
        coordinates,
        pairing_result.affine_realization,
    )
    _require_group_ring_boundary_zero(
        source_extension,
        d1_group,
        d2_group,
        edge_orbits,
        profile_edges,
        boundary_direction,
        coordinates,
        pairing_result.affine_realization,
    )

    chain = ChainComplexValue(
        coefficient_ring=CoefficientRing.INTEGER,
        prime=None,
        degree_min=0,
        degree_max=2,
        basis_sizes=(len(orbit_roots), len(edge_orbits), 1),
        differential_matrices=(
            tuple(tuple(str(value) for value in row) for row in d1),
            tuple(tuple(str(value) for value in row) for row in d2),
        ),
    )
    if any(
        sum(d1[row][edge] * d2[edge][0] for edge in range(len(edge_orbits)))
        for row in range(len(orbit_roots))
    ):
        _domain(
            "cellular_boundary_not_square_zero",
            "induced quotient cellular boundaries fail d1 composed with d2 equals zero",
            ("source", "pairings"),
        )
    return BieberbachFaceOrbitComplex(
        source=checked,
        vertex_orbits=tuple(tuple(vertices) for vertices in orbit_members.values()),
        edge_orbit_representatives=tuple(edge_orbits),
        orbit_maps=tuple(maps),
        boundary_1_to_0=tuple(d1_group),
        boundary_2_to_1=tuple(d2_group),
        quotient_chain_complex=chain,
    )


__all__ = ["quotient_face_orbit_complex"]
