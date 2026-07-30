"""Independent finite-graph replay with no producer or numeric-backend imports."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from functools import cache
from itertools import combinations, pairwise
from typing import Any

from jacobian_checkers.bound_artifacts import bound_request


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _accept(detail: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _accept_exhaustive(detail: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "EXHAUSTIVE_FINITE",
        "coverage": "EXHAUSTIVE",
        "detail": detail,
    }


def _run(
    request: object,
    *,
    operation_id: str,
    witness_format: str,
    replay_method: str,
    replay: Callable[[dict[str, Any], dict[str, Any]], bool],
    exhaustive: bool = False,
) -> dict[str, Any]:
    try:
        source, result = bound_request(
            request,
            operation_id=operation_id,
            witness_format=witness_format,
        )
        if not replay(source, result):
            return _reject(
                f"declared result does not match independent {replay_method}"
            )
        detail = f"independent {replay_method} accepted {operation_id}"
        return _accept_exhaustive(detail) if exhaustive else _accept(detail)
    except (KeyError, TypeError, ValueError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


def _finite_simple_graph(
    source: dict[str, Any],
    *,
    maximum_order: int,
) -> tuple[tuple[str, ...], set[tuple[str, str]], dict[str, set[str]]]:
    graph = source.get("graph")
    if not isinstance(graph, dict) or set(graph) != {
        "graph_schema_version",
        "vertices",
        "edges",
    }:
        raise ValueError("graph input is malformed")
    vertices = graph["vertices"]
    edges = graph["edges"]
    if (
        graph["graph_schema_version"] != "1"
        or not isinstance(vertices, list)
        or len(vertices) > maximum_order
        or not all(
            isinstance(vertex, str) and 0 < len(vertex) <= 256 for vertex in vertices
        )
        or len(vertices) != len(set(vertices))
        or not isinstance(edges, list)
        or len(edges) > len(vertices) * (len(vertices) - 1) // 2
    ):
        raise ValueError("graph lies outside the checker scope")
    vertex_set = set(vertices)
    normalized_edges: set[tuple[str, str]] = set()
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or not all(isinstance(endpoint, str) for endpoint in edge)
            or edge[0] == edge[1]
            or edge[0] not in vertex_set
            or edge[1] not in vertex_set
        ):
            raise ValueError("graph edge payload is malformed")
        normalized_edges.add(tuple(sorted((edge[0], edge[1]))))
    if len(normalized_edges) != len(edges):
        raise ValueError("graph edge payload contains duplicates")
    adjacency = {vertex: set[str]() for vertex in vertices}
    for left, right in normalized_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return tuple(vertices), normalized_edges, adjacency


def _all_sources_distance_rows(
    vertices: tuple[str, ...],
    adjacency: dict[str, set[str]],
) -> tuple[tuple[int | None, ...], ...]:
    rows: list[tuple[int | None, ...]] = []
    for source in vertices:
        distances = {source: 0}
        frontier = deque([source])
        while frontier:
            current = frontier.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    frontier.append(neighbor)
        rows.append(tuple(distances.get(target) for target in vertices))
    return tuple(rows)


def _canonical_edge(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _orbit_partition(
    elements: tuple[Any, ...],
    actions: tuple[dict[Any, Any], ...],
) -> tuple[tuple[Any, ...], ...]:
    parent = {element: element for element in elements}

    def find(element: Any) -> Any:
        root = element
        while parent[root] != root:
            root = parent[root]
        while parent[element] != element:
            next_element = parent[element]
            parent[element] = root
            element = next_element
        return root

    def union(left: Any, right: Any) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for action in actions:
        for element in elements:
            union(element, action[element])
    members_by_root: dict[Any, list[Any]] = {}
    for element in elements:
        members_by_root.setdefault(find(element), []).append(element)
    return tuple(
        sorted(
            (tuple(sorted(members)) for members in members_by_root.values()),
            key=lambda orbit: orbit[0],
        )
    )


def _graph_symmetry_generator_orbits(
    source: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    if (
        set(source)
        != {
            "graph",
            "generators",
            "vertex_colors",
            "edge_colors",
            "action",
        }
        or source["action"] != "DECLARED_AUTOMORPHISM_GENERATORS"
    ):
        return False
    vertices, normalized_edges, _ = _finite_simple_graph(
        source,
        maximum_order=256,
    )
    raw_graph = source["graph"]
    raw_edges = raw_graph["edges"]
    edges = tuple((edge[0], edge[1]) for edge in raw_edges)
    if (
        len(edges) > 4_096
        or any(not 0 < len(vertex) <= 64 for vertex in vertices)
        or set(edges) != normalized_edges
    ):
        return False

    raw_vertex_colors = source["vertex_colors"]
    if not isinstance(raw_vertex_colors, list) or len(raw_vertex_colors) not in {
        0,
        len(vertices),
    }:
        return False
    if raw_vertex_colors:
        if any(
            not isinstance(item, dict)
            or set(item) != {"vertex", "color"}
            or item["vertex"] != vertices[index]
            or not isinstance(item["color"], str)
            or not 0 < len(item["color"]) <= 128
            for index, item in enumerate(raw_vertex_colors)
        ):
            return False
        vertex_colors = {item["vertex"]: item["color"] for item in raw_vertex_colors}
    else:
        vertex_colors = dict.fromkeys(vertices, "__UNCOLORED__")

    raw_edge_colors = source["edge_colors"]
    if not isinstance(raw_edge_colors, list) or len(raw_edge_colors) not in {
        0,
        len(edges),
    }:
        return False
    if raw_edge_colors:
        if any(
            not isinstance(item, dict)
            or set(item) != {"edge", "color"}
            or item["edge"] != list(edges[index])
            or not isinstance(item["color"], str)
            or not 0 < len(item["color"]) <= 128
            for index, item in enumerate(raw_edge_colors)
        ):
            return False
        edge_colors = {
            (item["edge"][0], item["edge"][1]): item["color"]
            for item in raw_edge_colors
        }
    else:
        edge_colors = dict.fromkeys(edges, "__UNCOLORED__")

    raw_generators = source["generators"]
    if not isinstance(raw_generators, list) or len(raw_generators) > 64:
        return False
    vertex_set = set(vertices)
    generator_ids: list[str] = []
    vertex_actions: list[dict[str, str]] = []
    edge_actions: list[dict[tuple[str, str], tuple[str, str]]] = []
    for generator in raw_generators:
        if not isinstance(generator, dict) or set(generator) != {
            "generator_id",
            "mapping",
        }:
            return False
        generator_id = generator["generator_id"]
        mapping = generator["mapping"]
        if (
            not isinstance(generator_id, str)
            or not 0 < len(generator_id) <= 64
            or not isinstance(mapping, dict)
            or set(mapping) != vertex_set
            or set(mapping.values()) != vertex_set
            or any(
                not isinstance(source_vertex, str) or not isinstance(target_vertex, str)
                for source_vertex, target_vertex in mapping.items()
            )
        ):
            return False
        if any(
            vertex_colors[vertex] != vertex_colors[mapping[vertex]]
            for vertex in vertices
        ):
            return False
        edge_action = {
            edge: _canonical_edge(mapping[edge[0]], mapping[edge[1]]) for edge in edges
        }
        if set(edge_action.values()) != normalized_edges or any(
            edge_colors[edge] != edge_colors[edge_action[edge]] for edge in edges
        ):
            return False
        generator_ids.append(generator_id)
        vertex_actions.append(mapping)
        edge_actions.append(edge_action)
    if len(set(generator_ids)) != len(generator_ids):
        return False

    vertex_orbits = _orbit_partition(vertices, tuple(vertex_actions))
    edge_orbits = _orbit_partition(edges, tuple(edge_actions))
    expected = {
        "vertices": sorted(vertices),
        "edges": [list(edge) for edge in sorted(edges)],
        "generator_ids": sorted(generator_ids),
        "generator_count": len(generator_ids),
        "vertex_orbits": [
            {
                "orbit_index": index,
                "representative": members[0],
                "members": list(members),
            }
            for index, members in enumerate(vertex_orbits)
        ],
        "edge_orbits": [
            {
                "orbit_index": index,
                "representative": list(members[0]),
                "members": [list(edge) for edge in members],
            }
            for index, members in enumerate(edge_orbits)
        ],
        "vertex_orbit_count": len(vertex_orbits),
        "edge_orbit_count": len(edge_orbits),
        "vertex_color_mode": "DECLARED" if raw_vertex_colors else "UNCOLORED",
        "edge_color_mode": "DECLARED" if raw_edge_colors else "UNCOLORED",
        "action": "DECLARED_GENERATED_SUBGROUP",
        "generator_validation": ("ALL_DECLARED_GENERATORS_PRESERVE_GRAPH_AND_COLORS"),
        "orbit_completeness": "COMPLETE_FOR_DECLARED_GENERATORS",
        "automorphism_group_completeness": ("FULL_AUTOMORPHISM_GROUP_NOT_CLAIMED"),
        "exactness": "EXACT_COMBINATORIAL",
        "determinism": "DETERMINISTIC",
        "backend": "jacobian-stdlib",
        "backend_version": "1",
        "verification": "UNVERIFIED",
    }
    return result == expected


def check_graph_symmetry_generator_orbits(
    request: dict[str, Any],
) -> dict[str, Any]:
    return _run(
        request,
        operation_id="graph.symmetry.generator_orbits.compute",
        witness_format="graph.symmetry.generator-orbits.stdlib-replay",
        replay=_graph_symmetry_generator_orbits,
        replay_method="declared color-preserving generator orbit replay",
        exhaustive=True,
    )


def _all_sources_eccentricities(
    vertices: tuple[str, ...],
    adjacency: dict[str, set[str]],
) -> tuple[int, ...] | None:
    if not vertices:
        return None
    eccentricities: list[int] = []
    for row in _all_sources_distance_rows(vertices, adjacency):
        finite_row = tuple(distance for distance in row if distance is not None)
        if len(finite_row) != len(vertices):
            return None
        eccentricities.append(max(finite_row))
    return tuple(eccentricities)


def _graph_metric(
    source: dict[str, Any],
    result: dict[str, Any],
    *,
    field: str,
    inapplicable_detail: str,
    aggregate: Callable[[tuple[int, ...]], int],
) -> bool:
    vertices, _, adjacency = _finite_simple_graph(source, maximum_order=32)
    if set(result) != {
        "status",
        field,
        "connected",
        "exactness",
        "detail",
    }:
        return False
    eccentricities = _all_sources_eccentricities(vertices, adjacency)
    if eccentricities is None:
        return (
            result["status"] == "NOT_APPLICABLE"
            and result[field] is None
            and result["connected"] is False
            and result["exactness"] == "NOT_APPLICABLE"
            and result["detail"] == inapplicable_detail
        )
    claimed = result[field]
    return (
        result["status"] == "COMPUTED"
        and type(claimed) is int
        and claimed == aggregate(eccentricities)
        and result["connected"] is True
        and result["exactness"] == "EXACT"
        and result["detail"] is None
    )


def _diameter(source: dict[str, Any], result: dict[str, Any]) -> bool:
    return _graph_metric(
        source,
        result,
        field="diameter",
        inapplicable_detail="diameter requires a nonempty connected graph",
        aggregate=max,
    )


def check_graph_diameter(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="graph.invariant.diameter.compute",
        witness_format="graph.diameter.all-sources-bfs-v1",
        replay=_diameter,
        replay_method="all-sources breadth-first replay",
        exhaustive=True,
    )


def _radius(source: dict[str, Any], result: dict[str, Any]) -> bool:
    return _graph_metric(
        source,
        result,
        field="radius",
        inapplicable_detail="radius requires a nonempty connected graph",
        aggregate=min,
    )


def check_graph_radius(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="graph.invariant.radius.compute",
        witness_format="graph.radius.all-sources-bfs-v1",
        replay=_radius,
        replay_method="all-sources breadth-first replay",
        exhaustive=True,
    )


def _distance_matrix(source: dict[str, Any], result: dict[str, Any]) -> bool:
    input_vertices, normalized_edges, adjacency = _finite_simple_graph(
        source,
        maximum_order=32,
    )
    vertices = tuple(sorted(input_vertices))
    if (
        set(result)
        != {
            "semantics_version",
            "vertex_ordering",
            "pair_coverage",
            "unreachable_representation",
            "vertices",
            "distances",
            "connected",
        }
        or result["semantics_version"] != "unweighted-shortest-path-distance-matrix.v1"
        or result["vertex_ordering"] != "LEXICOGRAPHIC_ASCENDING"
        or result["pair_coverage"] != "ALL_ORDERED_VERTEX_PAIRS"
        or result["unreachable_representation"] != "JSON_NULL"
        or result["vertices"] != list(vertices)
        or type(result["connected"]) is not bool
    ):
        return False

    matrix = result["distances"]
    order = len(vertices)
    if (
        not isinstance(matrix, list)
        or len(matrix) != order
        or any(not isinstance(row, list) or len(row) != order for row in matrix)
    ):
        return False
    for source_index, row in enumerate(matrix):
        for target_index, distance in enumerate(row):
            if distance is not None and (
                type(distance) is not int or distance < 0 or distance > 31
            ):
                return False
            if source_index == target_index:
                if distance != 0:
                    return False
            elif distance == 0:
                return False
            if distance != matrix[target_index][source_index]:
                return False

    vertex_indices = {vertex: index for index, vertex in enumerate(vertices)}
    if any(
        matrix[vertex_indices[left]][vertex_indices[right]] != 1
        for left, right in normalized_edges
    ):
        return False
    for source_index in range(order):
        for intermediate_index in range(order):
            left = matrix[source_index][intermediate_index]
            if left is None:
                continue
            for target_index in range(order):
                right = matrix[intermediate_index][target_index]
                if right is None:
                    continue
                direct = matrix[source_index][target_index]
                if direct is None or direct > left + right:
                    return False

    expected = _all_sources_distance_rows(vertices, adjacency)
    expected_connected = bool(vertices) and all(
        distance is not None for row in expected for distance in row
    )
    return matrix == [list(row) for row in expected] and (
        result["connected"] is expected_connected
    )


def check_graph_distance_matrix(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="graph.distance_matrix.compute",
        witness_format="graph.distance-matrix.all-sources-bfs-v1",
        replay=_distance_matrix,
        replay_method="all-sources breadth-first distance-matrix replay",
        exhaustive=True,
    )


def _induced_tree_maximum(
    source: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    vertices, normalized_edges, adjacency = _finite_simple_graph(
        source,
        maximum_order=16,
    )
    vertex_set = set(vertices)

    def is_induced_tree(candidate: tuple[str, ...]) -> bool:
        if not candidate:
            return False
        selected = set(candidate)
        edge_count = sum(
            1
            for left, right in normalized_edges
            if left in selected and right in selected
        )
        if edge_count != len(candidate) - 1:
            return False
        reached = {candidate[0]}
        frontier = [candidate[0]]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current] & selected:
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        return len(reached) == len(candidate)

    claimed = result.get("optimum_value")
    witness = result.get("witness_vertices")
    if (
        result.get("status") != "EXACT"
        or result.get("convention") != "NONEMPTY_CONNECTED_ACYCLIC_EMPTY_SOURCE_ZERO"
        or type(claimed) is not int
        or claimed < 0
        or claimed > len(vertices)
        or result.get("order") != len(vertices)
        or result.get("incumbent_value") != claimed
        or result.get("lower_bound") != claimed
        or result.get("upper_bound") != claimed
        or not isinstance(witness, list)
        or len(witness) != claimed
        or not all(isinstance(vertex, str) for vertex in witness)
        or len(witness) != len(set(witness))
        or any(vertex not in vertex_set for vertex in witness)
    ):
        return False
    if claimed == 0:
        if vertices or witness:
            return False
    elif not is_induced_tree(tuple(witness)):
        return False

    actual = 0
    for cardinality in range(len(vertices), 0, -1):
        if any(
            is_induced_tree(candidate)
            for candidate in combinations(vertices, cardinality)
        ):
            actual = cardinality
            break
    return actual == claimed


def check_graph_induced_tree_maximum(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="graph.induced_tree.maximum.compute",
        witness_format="graph.induced-tree.maximum.exhaustive-replay",
        replay=_induced_tree_maximum,
        replay_method="finite-subset exhaustive replay",
    )


def _hamiltonian_path(
    source: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    vertices, normalized_edges, adjacency = _finite_simple_graph(
        source,
        maximum_order=18,
    )
    if set(result) != {
        "result_schema_version",
        "decision",
        "order",
        "path",
        "convention",
        "completion",
        "verification_capability_id",
        "verification_input_field",
    }:
        return False
    if (
        result["result_schema_version"] != "1"
        or result["order"] != len(vertices)
        or result["convention"] != "EMPTY_GRAPH_HAS_EMPTY_HAMILTONIAN_PATH"
        or result["completion"] != "COMPLETE"
        or result["verification_capability_id"] != "graph.hamiltonian_path.verify"
        or result["verification_input_field"] != "result_uri"
        or not isinstance(result["path"], list)
        or not all(isinstance(vertex, str) for vertex in result["path"])
    ):
        return False
    decision = result["decision"]
    path = result["path"]
    vertex_set = set(vertices)
    if decision == "EXISTS":
        return (
            len(path) == len(vertices)
            and len(path) == len(set(path))
            and set(path) == vertex_set
            and all(
                tuple(sorted((left, right))) in normalized_edges
                for left, right in pairwise(path)
            )
        )
    if decision != "DOES_NOT_EXIST" or path:
        return False
    if not vertices:
        return False

    index = {vertex: position for position, vertex in enumerate(vertices)}
    adjacency_masks = tuple(
        sum(1 << index[neighbor] for neighbor in adjacency[vertex])
        for vertex in vertices
    )
    full_mask = (1 << len(vertices)) - 1

    @cache
    def can_finish(last: int, visited: int) -> bool:
        if visited == full_mask:
            return True
        available = adjacency_masks[last] & ~visited
        while available:
            bit = available & -available
            if can_finish(bit.bit_length() - 1, visited | bit):
                return True
            available ^= bit
        return False

    exists = any(can_finish(start, 1 << start) for start in range(len(vertices)))
    return not exists


def check_graph_hamiltonian_path(request: dict[str, Any]) -> dict[str, Any]:
    try:
        source, result = bound_request(
            request,
            operation_id="graph.hamiltonian_path.decide",
            witness_format="graph.hamiltonian-path.exhaustive-replay",
        )
        if not _hamiltonian_path(source, result):
            return _reject(
                "declared result does not match independent finite "
                "Hamiltonian-path replay"
            )
        detail = (
            "independent finite Hamiltonian-path replay accepted "
            "graph.hamiltonian_path.decide"
        )
        return (
            _accept_exhaustive(detail)
            if result.get("decision") == "DOES_NOT_EXIST"
            else _accept(detail)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


def _maximum_matching(
    source: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    vertices, normalized_edges, adjacency = _finite_simple_graph(
        source,
        maximum_order=32,
    )
    if set(result) != {
        "maximum_matching_cardinality",
        "witness_edges",
        "certificate",
    }:
        return False
    claimed = result["maximum_matching_cardinality"]
    witness = result["witness_edges"]
    if (
        type(claimed) is not int
        or claimed < 0
        or claimed > len(vertices) // 2
        or not isinstance(witness, list)
        or len(witness) != claimed
    ):
        return False
    parsed_witness: list[tuple[str, str]] = []
    used_vertices: set[str] = set()
    for edge in witness:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or not all(isinstance(endpoint, str) for endpoint in edge)
            or edge[0] >= edge[1]
            or (edge[0], edge[1]) not in normalized_edges
            or edge[0] in used_vertices
            or edge[1] in used_vertices
        ):
            return False
        parsed_witness.append((edge[0], edge[1]))
        used_vertices.update(edge)
    if parsed_witness != sorted(parsed_witness):
        return False

    certificate = result["certificate"]
    if not isinstance(certificate, dict) or set(certificate) != {
        "certificate_schema_version",
        "kind",
        "barrier_vertices",
        "odd_component_count",
        "upper_bound",
    }:
        return False
    barrier = certificate["barrier_vertices"]
    declared_odd_components = certificate["odd_component_count"]
    declared_upper_bound = certificate["upper_bound"]
    vertex_set = set(vertices)
    if (
        certificate["certificate_schema_version"] != "1"
        or certificate["kind"] != "TUTTE_BERGE_BARRIER"
        or not isinstance(barrier, list)
        or not all(isinstance(vertex, str) for vertex in barrier)
        or barrier != sorted(set(barrier))
        or any(vertex not in vertex_set for vertex in barrier)
        or type(declared_odd_components) is not int
        or type(declared_upper_bound) is not int
        or declared_odd_components < 0
        or declared_upper_bound < 0
    ):
        return False

    barrier_set = set(barrier)
    unseen = vertex_set - barrier_set
    odd_component_count = 0
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        component = {first}
        frontier = [first]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current] - barrier_set:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        odd_component_count += len(component) % 2

    numerator = len(vertices) + len(barrier) - odd_component_count
    if numerator < 0 or numerator % 2:
        return False
    upper_bound = numerator // 2
    return (
        declared_odd_components == odd_component_count
        and declared_upper_bound == upper_bound
        and claimed == upper_bound
    )


def check_graph_maximum_matching(request: dict[str, Any]) -> dict[str, Any]:
    return _run(
        request,
        operation_id="graph.invariant.maximum_matching.compute",
        witness_format="graph.maximum-matching.tutte-berge-v1",
        replay=_maximum_matching,
        replay_method="Tutte-Berge barrier replay",
    )


__all__ = [
    "check_graph_diameter",
    "check_graph_distance_matrix",
    "check_graph_hamiltonian_path",
    "check_graph_induced_tree_maximum",
    "check_graph_maximum_matching",
    "check_graph_radius",
    "check_graph_symmetry_generator_orbits",
]
