"""Independent finite-graph replay with no producer or numeric-backend imports."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable
from fractions import Fraction
from functools import cache
from itertools import combinations, pairwise
from typing import Any, cast

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


def _parse_graph_rational(payload: object) -> Fraction:
    if not isinstance(payload, dict) or set(payload) != {"num", "den"}:
        raise ValueError("graph weight is malformed")
    numerator = payload["num"]
    denominator = payload["den"]
    if (
        not isinstance(numerator, str)
        or not isinstance(denominator, str)
        or len(numerator.lstrip("-")) > 256
        or len(denominator) > 256
        or re.fullmatch(r"(?:0|-?[1-9][0-9]*)", numerator) is None
        or re.fullmatch(r"[1-9][0-9]*", denominator) is None
    ):
        raise ValueError("graph weight lies outside the checker scope")
    value = Fraction(int(numerator), int(denominator))
    if str(value.numerator) != numerator or str(value.denominator) != denominator:
        raise ValueError("graph weight is not canonical")
    return value


def _parse_weighted_edge(
    raw_edge: object,
    vertex_set: set[str],
) -> tuple[tuple[str, str], Fraction]:
    if not isinstance(raw_edge, dict) or set(raw_edge) != {"endpoints", "weight"}:
        raise ValueError("weighted edge is malformed")
    endpoints = raw_edge["endpoints"]
    if (
        not isinstance(endpoints, list)
        or len(endpoints) != 2
        or not all(isinstance(endpoint, str) for endpoint in endpoints)
        or endpoints[0] == endpoints[1]
        or endpoints[0] not in vertex_set
        or endpoints[1] not in vertex_set
    ):
        raise ValueError("weighted edge endpoints are malformed")
    return _canonical_edge(endpoints[0], endpoints[1]), _parse_graph_rational(
        raw_edge["weight"]
    )


def _finite_weighted_graph(
    source: dict[str, Any],
) -> tuple[
    tuple[str, ...],
    dict[tuple[str, str], Fraction],
    dict[str, set[str]],
]:
    if set(source) != {"graph"}:
        raise ValueError("weighted graph request is malformed")
    graph = source["graph"]
    if not isinstance(graph, dict) or set(graph) != {
        "weighted_graph_schema_version",
        "vertices",
        "edges",
    }:
        raise ValueError("weighted graph input is malformed")
    vertices = graph["vertices"]
    raw_edges = graph["edges"]
    if (
        graph["weighted_graph_schema_version"] != "1"
        or not isinstance(vertices, list)
        or len(vertices) > 32
        or len(vertices) != len(set(vertices))
        or not all(
            isinstance(vertex, str) and 0 < len(vertex) <= 256 for vertex in vertices
        )
        or not isinstance(raw_edges, list)
        or len(raw_edges) > len(vertices) * (len(vertices) - 1) // 2
    ):
        raise ValueError("weighted graph lies outside the checker scope")
    vertex_set = set(vertices)
    weights = dict(_parse_weighted_edge(raw_edge, vertex_set) for raw_edge in raw_edges)
    if len(weights) != len(raw_edges):
        raise ValueError("weighted graph contains parallel undirected edges")
    adjacency = {vertex: set[str]() for vertex in vertices}
    for left, right in weights:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return tuple(vertices), weights, adjacency


def _component_partition(
    vertices: tuple[str, ...],
    adjacency: dict[str, set[str]],
) -> tuple[tuple[str, ...], ...]:
    unseen = set(vertices)
    components: list[tuple[str, ...]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        component = {root}
        frontier = [root]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current] & unseen:
                unseen.remove(neighbor)
                component.add(neighbor)
                frontier.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda component: component[0]))


def _fraction_payload(value: Fraction) -> dict[str, str]:
    return {"num": str(value.numerator), "den": str(value.denominator)}


def _no_spanning_tree_payload(
    vertices: tuple[str, ...],
    components: tuple[tuple[str, ...], ...],
) -> dict[str, Any]:
    return {
        "result_schema_version": "1",
        "status": "NO_SPANNING_TREE",
        "vertices": sorted(vertices),
        "order": len(vertices),
        "connected": False,
        "component_count": len(components),
        "components": [list(component) for component in components],
        "tree_edges": [],
        "total_weight": None,
        "optimality_certificate": {
            "certificate_schema_version": "1",
            "method": "ALL_FUNDAMENTAL_CYCLES_NON_IMPROVING",
            "checks": [],
            "required_checks": [
                "SOURCE_CONNECTIVITY",
                "TREE_SPANNING_ACYCLIC",
                "TOTAL_WEIGHT_EXACT",
                "ALL_NON_TREE_EDGES_COVERED",
                "CYCLE_NON_IMPROVEMENT",
            ],
        },
        "convention": (
            "MINIMUM_TOTAL_EDGE_WEIGHT_OVER_QQ_EMPTY_GRAPH_HAS_NO_SPANNING_TREE"
        ),
        "completion": "COMPLETE",
    }


def _parse_tree_edges(
    raw_edges: object,
    source_weights: dict[tuple[str, str], Fraction],
    order: int,
) -> tuple[
    dict[tuple[str, str], Fraction],
    dict[str, dict[str, Fraction]],
]:
    if not isinstance(raw_edges, list) or len(raw_edges) != order - 1:
        raise ValueError("declared tree has the wrong edge count")
    tree_weights: dict[tuple[str, str], Fraction] = {}
    declared_endpoints: list[tuple[str, str]] = []
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, dict) or set(raw_edge) != {
            "endpoints",
            "weight",
        }:
            raise ValueError("declared tree edge is malformed")
        endpoints = raw_edge["endpoints"]
        if (
            not isinstance(endpoints, list)
            or len(endpoints) != 2
            or not all(isinstance(endpoint, str) for endpoint in endpoints)
            or endpoints[0] >= endpoints[1]
        ):
            raise ValueError("declared tree edge orientation is not canonical")
        edge = (endpoints[0], endpoints[1])
        if edge not in source_weights or raw_edge["weight"] != _fraction_payload(
            source_weights[edge]
        ):
            raise ValueError("declared tree edge does not match the source graph")
        declared_endpoints.append(edge)
        tree_weights[edge] = source_weights[edge]
    if declared_endpoints != sorted(declared_endpoints) or len(tree_weights) != len(
        raw_edges
    ):
        raise ValueError("declared tree edges are not unique and canonical")
    adjacency: dict[str, dict[str, Fraction]] = {}
    for (left, right), weight in tree_weights.items():
        adjacency.setdefault(left, {})[right] = weight
        adjacency.setdefault(right, {})[left] = weight
    return tree_weights, adjacency


def _tree_reaches_every_vertex(
    vertices: tuple[str, ...],
    adjacency: dict[str, dict[str, Fraction]],
) -> bool:
    reached = {vertices[0]}
    frontier = [vertices[0]]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency.get(current, {}):
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    return reached == set(vertices)


def _tree_path(
    adjacency: dict[str, dict[str, Fraction]],
    source: str,
    target: str,
) -> tuple[str, ...]:
    predecessor: dict[str, str | None] = {source: None}
    frontier = deque([source])
    while frontier and target not in predecessor:
        current = frontier.popleft()
        for neighbor in sorted(adjacency.get(current, {})):
            if neighbor not in predecessor:
                predecessor[neighbor] = current
                frontier.append(neighbor)
    if target not in predecessor:
        raise ValueError("declared tree does not join a source edge's endpoints")
    reversed_path = [target]
    while predecessor[reversed_path[-1]] is not None:
        reversed_path.append(cast(str, predecessor[reversed_path[-1]]))
    return tuple(reversed(reversed_path))


def _expected_mst_certificate(
    source_weights: dict[tuple[str, str], Fraction],
    tree_weights: dict[tuple[str, str], Fraction],
    tree_adjacency: dict[str, dict[str, Fraction]],
) -> dict[str, Any] | None:
    checks: list[dict[str, Any]] = []
    for edge in sorted(set(source_weights) - set(tree_weights)):
        path = _tree_path(tree_adjacency, *edge)
        maximum = max(
            tree_adjacency[path[index]][path[index + 1]]
            for index in range(len(path) - 1)
        )
        if source_weights[edge] < maximum:
            return None
        checks.append(
            {
                "non_tree_edge": list(edge),
                "edge_weight": _fraction_payload(source_weights[edge]),
                "tree_path_vertices": list(path),
                "maximum_tree_path_weight": _fraction_payload(maximum),
                "condition": "EDGE_WEIGHT_GTE_MAXIMUM_TREE_PATH_WEIGHT",
            }
        )
    return {
        "certificate_schema_version": "1",
        "method": "ALL_FUNDAMENTAL_CYCLES_NON_IMPROVING",
        "checks": checks,
        "required_checks": [
            "SOURCE_CONNECTIVITY",
            "TREE_SPANNING_ACYCLIC",
            "TOTAL_WEIGHT_EXACT",
            "ALL_NON_TREE_EDGES_COVERED",
            "CYCLE_NON_IMPROVEMENT",
        ],
    }


def _connected_mst_result(
    result: dict[str, Any],
    *,
    vertices: tuple[str, ...],
    components: tuple[tuple[str, ...], ...],
    source_weights: dict[tuple[str, str], Fraction],
) -> bool:
    if (
        set(result)
        != {
            "result_schema_version",
            "status",
            "vertices",
            "order",
            "connected",
            "component_count",
            "components",
            "tree_edges",
            "total_weight",
            "optimality_certificate",
            "convention",
            "completion",
        }
        or result["result_schema_version"] != "1"
        or result["status"] != "EXACT"
        or result["vertices"] != sorted(vertices)
        or result["order"] != len(vertices)
        or result["connected"] is not True
        or result["component_count"] != 1
        or result["components"] != [list(component) for component in components]
        or result["convention"]
        != ("MINIMUM_TOTAL_EDGE_WEIGHT_OVER_QQ_EMPTY_GRAPH_HAS_NO_SPANNING_TREE")
        or result["completion"] != "COMPLETE"
    ):
        return False
    tree_weights, tree_adjacency = _parse_tree_edges(
        result["tree_edges"],
        source_weights,
        len(vertices),
    )
    if not _tree_reaches_every_vertex(vertices, tree_adjacency):
        return False
    if result["total_weight"] != _fraction_payload(
        sum(tree_weights.values(), start=Fraction())
    ):
        return False
    certificate = _expected_mst_certificate(
        source_weights,
        tree_weights,
        tree_adjacency,
    )
    return certificate is not None and result["optimality_certificate"] == certificate


def _minimum_spanning_tree(
    source: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    vertices, source_weights, adjacency = _finite_weighted_graph(source)
    components = _component_partition(vertices, adjacency)
    if not vertices or len(components) != 1:
        return result == _no_spanning_tree_payload(vertices, components)
    return _connected_mst_result(
        result,
        vertices=vertices,
        components=components,
        source_weights=source_weights,
    )


def _mst_decision(
    *,
    accepted: bool,
    detail: str,
    disconnected: bool = False,
) -> dict[str, Any]:
    return {
        "accepted": accepted,
        "conclusion": "TRUE" if accepted else "UNKNOWN",
        "arithmetic": "EXACT_RATIONAL",
        "method": "EXHAUSTIVE_FINITE" if disconnected else "CHECKED_CERTIFICATE",
        "coverage": "EXHAUSTIVE" if disconnected else "NOT_APPLICABLE",
        "detail": detail,
    }


def check_graph_minimum_spanning_tree(
    request: dict[str, Any],
) -> dict[str, Any]:
    try:
        source, result = bound_request(
            request,
            operation_id="graph.spanning_tree.minimum.compute",
            witness_format="graph.minimum-spanning-tree.cycle-certificate-v1",
        )
        if not _minimum_spanning_tree(source, result):
            return _mst_decision(
                accepted=False,
                detail=(
                    "declared result does not match independent exact rational "
                    "spanning-tree and cycle-certificate replay"
                ),
            )
        disconnected = result.get("status") == "NO_SPANNING_TREE"
        return _mst_decision(
            accepted=True,
            disconnected=disconnected,
            detail=(
                "independent finite connectivity replay accepted "
                "graph.spanning_tree.minimum.compute"
                if disconnected
                else (
                    "independent fundamental-cycle optimality certificate replay "
                    "accepted graph.spanning_tree.minimum.compute"
                )
            ),
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return _mst_decision(
            accepted=False,
            detail="malformed, unsupported, or mismatched checker request",
        )


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


def _parse_symmetry_vertex_colors(
    raw_vertex_colors: object,
    vertices: tuple[str, ...],
) -> dict[str, str] | None:
    if not isinstance(raw_vertex_colors, list) or len(raw_vertex_colors) not in {
        0,
        len(vertices),
    }:
        return None
    if raw_vertex_colors:
        if any(
            not isinstance(item, dict)
            or set(item) != {"vertex", "color"}
            or item["vertex"] != vertices[index]
            or not isinstance(item["color"], str)
            or not 0 < len(item["color"]) <= 128
            for index, item in enumerate(raw_vertex_colors)
        ):
            return None
        return {item["vertex"]: item["color"] for item in raw_vertex_colors}
    return dict.fromkeys(vertices, "__UNCOLORED__")


def _parse_symmetry_edge_colors(
    raw_edge_colors: object,
    edges: tuple[tuple[str, str], ...],
) -> dict[tuple[str, str], str] | None:
    if not isinstance(raw_edge_colors, list) or len(raw_edge_colors) not in {
        0,
        len(edges),
    }:
        return None
    if raw_edge_colors:
        if any(
            not isinstance(item, dict)
            or set(item) != {"edge", "color"}
            or item["edge"] != list(edges[index])
            or not isinstance(item["color"], str)
            or not 0 < len(item["color"]) <= 128
            for index, item in enumerate(raw_edge_colors)
        ):
            return None
        return {
            (item["edge"][0], item["edge"][1]): item["color"]
            for item in raw_edge_colors
        }
    return dict.fromkeys(edges, "__UNCOLORED__")


def _validate_symmetry_generator(
    generator: object,
    *,
    vertices: tuple[str, ...],
    vertex_set: set[str],
    edges: tuple[tuple[str, str], ...],
    normalized_edges: set[tuple[str, str]],
    vertex_colors: dict[str, str],
    edge_colors: dict[tuple[str, str], str],
) -> tuple[str, dict[str, str], dict[tuple[str, str], tuple[str, str]]] | None:
    if not isinstance(generator, dict) or set(generator) != {
        "generator_id",
        "mapping",
    }:
        return None
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
        return None
    if any(
        vertex_colors[vertex] != vertex_colors[mapping[vertex]] for vertex in vertices
    ):
        return None
    edge_action = {
        edge: _canonical_edge(mapping[edge[0]], mapping[edge[1]]) for edge in edges
    }
    if set(edge_action.values()) != normalized_edges or any(
        edge_colors[edge] != edge_colors[edge_action[edge]] for edge in edges
    ):
        return None
    return generator_id, mapping, edge_action


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

    vertex_colors = _parse_symmetry_vertex_colors(source["vertex_colors"], vertices)
    if vertex_colors is None:
        return False

    edge_colors = _parse_symmetry_edge_colors(source["edge_colors"], edges)
    if edge_colors is None:
        return False

    raw_generators = source["generators"]
    if not isinstance(raw_generators, list) or len(raw_generators) > 64:
        return False
    vertex_set = set(vertices)
    generator_ids: list[str] = []
    vertex_actions: list[dict[str, str]] = []
    edge_actions: list[dict[tuple[str, str], tuple[str, str]]] = []
    for generator in raw_generators:
        parsed = _validate_symmetry_generator(
            generator,
            vertices=vertices,
            vertex_set=vertex_set,
            edges=edges,
            normalized_edges=normalized_edges,
            vertex_colors=vertex_colors,
            edge_colors=edge_colors,
        )
        if parsed is None:
            return False
        generator_ids.append(parsed[0])
        vertex_actions.append(parsed[1])
        edge_actions.append(parsed[2])
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
        "vertex_color_mode": "DECLARED" if source["vertex_colors"] else "UNCOLORED",
        "edge_color_mode": "DECLARED" if source["edge_colors"] else "UNCOLORED",
        "action": "DECLARED_GENERATED_SUBGROUP",
        "generator_validation": ("ALL_DECLARED_GENERATORS_PRESERVE_GRAPH_AND_COLORS"),
        "orbit_completeness": "COMPLETE_FOR_DECLARED_GENERATORS",
        "automorphism_group_completeness": ("FULL_AUTOMORPHISM_GROUP_NOT_CLAIMED"),
        "exactness": "EXACT_COMBINATORIAL",
        "determinism": "DETERMINISTIC",
        "backend": "networkx",
        "backend_version": "3.6.1",
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


def _validate_distance_matrix_header(
    result: dict[str, Any],
    vertices: tuple[str, ...],
) -> bool:
    return not (
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
    )


def _validate_distance_matrix_entries(
    matrix: object,
    order: int,
) -> bool:
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
    return True


def _validate_distance_matrix_triangle(
    matrix: list[list[int | None]],
    order: int,
) -> bool:
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
    return True


def _distance_matrix(source: dict[str, Any], result: dict[str, Any]) -> bool:
    input_vertices, normalized_edges, adjacency = _finite_simple_graph(
        source,
        maximum_order=32,
    )
    vertices = tuple(sorted(input_vertices))
    if not _validate_distance_matrix_header(result, vertices):
        return False

    matrix = result["distances"]
    order = len(vertices)
    if not _validate_distance_matrix_entries(matrix, order):
        return False

    vertex_indices = {vertex: index for index, vertex in enumerate(vertices)}
    if any(
        matrix[vertex_indices[left]][vertex_indices[right]] != 1
        for left, right in normalized_edges
    ):
        return False
    if not _validate_distance_matrix_triangle(matrix, order):
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


def _is_induced_tree(
    candidate: tuple[str, ...],
    normalized_edges: set[tuple[str, str]],
    adjacency: dict[str, set[str]],
) -> bool:
    if not candidate:
        return False
    selected = set(candidate)
    edge_count = sum(
        1 for left, right in normalized_edges if left in selected and right in selected
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


def _validate_induced_tree_result(
    result: dict[str, Any],
    vertices: tuple[str, ...],
    vertex_set: set[str],
) -> bool:
    claimed = result.get("optimum_value")
    witness = result.get("witness_vertices")
    return not (
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
    if not _validate_induced_tree_result(result, vertices, vertex_set):
        return False
    claimed = result["optimum_value"]
    witness = result["witness_vertices"]
    if claimed == 0:
        if vertices or witness:
            return False
    elif not _is_induced_tree(tuple(witness), normalized_edges, adjacency):
        return False

    actual = 0
    for cardinality in range(len(vertices), 0, -1):
        if any(
            _is_induced_tree(candidate, normalized_edges, adjacency)
            for candidate in combinations(vertices, cardinality)
        ):
            actual = cardinality
            break
    return bool(actual == claimed)


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


def _parse_matching_witness(
    witness: object,
    claimed: int,
    normalized_edges: set[tuple[str, str]],
) -> list[tuple[str, str]] | None:
    if not isinstance(witness, list):
        return None
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
            return None
        parsed_witness.append((edge[0], edge[1]))
        used_vertices.update(edge)
    if parsed_witness != sorted(parsed_witness):
        return None
    return parsed_witness


def _validate_matching_certificate(
    certificate: object,
    vertex_set: set[str],
) -> tuple[list[str], int, int] | None:
    if not isinstance(certificate, dict) or set(certificate) != {
        "certificate_schema_version",
        "kind",
        "barrier_vertices",
        "odd_component_count",
        "upper_bound",
    }:
        return None
    barrier = certificate["barrier_vertices"]
    declared_odd_components = certificate["odd_component_count"]
    declared_upper_bound = certificate["upper_bound"]
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
        return None
    return barrier, declared_odd_components, declared_upper_bound


def _count_odd_components(
    vertex_set: set[str],
    barrier_set: set[str],
    adjacency: dict[str, set[str]],
) -> int:
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
    return odd_component_count


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
    parsed_witness = _parse_matching_witness(witness, claimed, normalized_edges)
    if parsed_witness is None:
        return False

    vertex_set = set(vertices)
    certificate_result = _validate_matching_certificate(
        result["certificate"], vertex_set
    )
    if certificate_result is None:
        return False
    barrier, declared_odd_components, declared_upper_bound = certificate_result

    barrier_set = set(barrier)
    odd_component_count = _count_odd_components(vertex_set, barrier_set, adjacency)

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
    "check_graph_minimum_spanning_tree",
    "check_graph_radius",
    "check_graph_symmetry_generator_orbits",
]
